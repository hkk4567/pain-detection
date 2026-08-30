# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from collections import Counter
import torchvision.transforms as transforms

# Cố gắng import từ file gốc (đảm bảo file gốc tên là RGB.py)
try:
    from RGB import (
        Config, BaselinePainModel, PainDataset, SplitTransformDataset,
        custom_collate, categorize_pain, PAIN_LABELS, SmartFaceCLAHE, seed_everything
    )
except ImportError:
    print("❌ Lỗi: Không tìm thấy file RGB.py hoặc lỗi import.")
    print("Vui lòng đặt file này cùng thư mục với RGB.py")
    exit(1)

# ==========================================
# CẤU HÌNH ĐÁNH GIÁ
# ==========================================
# TODO: Bạn CẦN thay đổi timestamp này thành timestamp của lần chạy RGB mà bạn muốn đánh giá
TARGET_RUN_TIMESTAMP = "2026-04-15_17-07-38_V11_RGB_Fairness" 

def main():
    cfg = Config()
    cfg.NUM_WORKERS = 0
    seed_everything(cfg.SEED)
    device = cfg.DEVICE

    print(f"🚀 Bắt đầu tạo Overall Confusion Matrix cho RUN: {TARGET_RUN_TIMESTAMP}")
    print(f"🖥️ Device: {device}")

    # 1. Tái tạo Eval Transform cho RGB
    clahe = SmartFaceCLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
    eval_transform = transforms.Compose([
        clahe,
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        # Đã bỏ biến đổi Grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Khởi tạo Dataset và tái tạo Split để lấy đúng tập Test
    print("📦 Đang load dataset...")
    full_dataset = PainDataset(cfg)
    eval_dataset = SplitTransformDataset(full_dataset, eval_transform)

    all_indices = np.arange(len(full_dataset))
    all_labels = np.array(full_dataset.stratify_labels)
    all_groups = np.array(full_dataset.groups)

    # Loại bỏ rare groups hệt như script train
    counts = Counter(all_labels)
    rare_groups = [g for g, c in counts.items() if c < 2]
    if rare_groups:
        valid_mask = np.isin(all_labels, rare_groups, invert=True)
        all_indices = all_indices[valid_mask]
        all_labels = all_labels[valid_mask]
        all_groups = all_groups[valid_mask]

    # =========================================================================
    # [CUSTOM SPLIT] TÁI TẠO TẬP TEST THEO TỶ LỆ CỐ ĐỊNH 1:2:3 (ASIAN:BLACK:WHITE)
    # =========================================================================
    subject_to_race = {}
    for idx in all_indices:
        subj = full_dataset.groups[idx]
        race = full_dataset.data_list[idx]['race']
        subject_to_race[subj] = race

    asian_subjects = [s for s, r in subject_to_race.items() if r == 'Asian']
    black_subjects = [s for s, r in subject_to_race.items() if r == 'Black']
    white_subjects = [s for s, r in subject_to_race.items() if r == 'White']

    # Xáo trộn danh sách Subject với cùng Seed cố định lúc huấn luyện
    rng = random.Random(cfg.SEED)
    rng.shuffle(asian_subjects)
    rng.shuffle(black_subjects)
    rng.shuffle(white_subjects)

    n_asian = 6
    n_black = 12
    n_white = 18

    test_subjects = set(
        asian_subjects[:n_asian] + 
        black_subjects[:n_black] + 
        white_subjects[:n_white]
    )

    test_idx_list = []
    for idx in all_indices:
        if full_dataset.groups[idx] in test_subjects:
            test_idx_list.append(idx)

    test_idx = np.array(test_idx_list)
    # =========================================================================

    test_loader = DataLoader(
        Subset(eval_dataset, test_idx),
        batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=custom_collate,
        num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    print(f"✅ Đã chuẩn bị Test Loader với {len(test_idx)} mẫu.")

    # 3. Load 10 Models
    ensemble_models = []
    for fold in range(cfg.K_FOLDS):
        cp_path = os.path.join("checkpoints", f"{TARGET_RUN_TIMESTAMP}_fold_{fold+1}.pth")
        if os.path.exists(cp_path):
            model = BaselinePainModel(cfg).to(device)
            model.load_state_dict(torch.load(cp_path, map_location=device, weights_only=True))
            model.eval()
            ensemble_models.append(model)
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy {cp_path}")

    if not ensemble_models:
        print("❌ Lỗi: Không có model nào được load. Hãy kiểm tra lại TARGET_RUN_TIMESTAMP.")
        return

    print(f"🧠 Đã load {len(ensemble_models)}/{cfg.K_FOLDS} models cho Ensemble.")

    # 4. Chạy Inference
    all_labels_test, all_preds_test = [], []
    print("⏳ Đang chạy dự đoán trên tập test...")
    with torch.no_grad():
        for batch_data in test_loader:
            if batch_data is None: continue
            imgs, labels, _ = batch_data
            imgs = imgs.to(device)
            
            # Lấy dự đoán từ tất cả các models và tính trung bình
            fold_preds = [m(imgs).squeeze(-1).cpu().numpy() for m in ensemble_models]
            ensemble_pred = np.mean(fold_preds, axis=0)
            
            all_labels_test.extend(labels.numpy())
            all_preds_test.extend(ensemble_pred.tolist())

    # 5. Phân loại và tính Confusion Matrix
    y_true_cat = [categorize_pain(x) for x in all_labels_test]
    y_pred_cat = [categorize_pain(x) for x in all_preds_test]

    _cm = confusion_matrix(y_true_cat, y_pred_cat, labels=[0, 1, 2])
    _rs = _cm.sum(axis=1, keepdims=True)
    _cm_recall = np.divide(_cm.astype(float), _rs, out=np.zeros_like(_cm, dtype=float), where=_rs != 0)

    # 6. Vẽ biểu đồ Confusion Matrix
    export_dir = os.path.join("exported_data", TARGET_RUN_TIMESTAMP)
    os.makedirs(export_dir, exist_ok=True)
    
    cm_fig, cm_axes = plt.subplots(1, 2, figsize=(13, 5))
    cm_fig.patch.set_facecolor("#1a1a2e")
    
    for ax, data, fmt, cmap, title in [
        (cm_axes[0], _cm, "d", sns.color_palette(["#2d2b55","#4a4080","#9b59b6","#ff79c6"], as_cmap=True), "Overall Count"),
        (cm_axes[1], _cm_recall, ".2f", "YlGn", "Overall Recall"),
    ]:
        ax.set_facecolor("#16213e")
        text_color = "white" if "Count" in title else "#1a1a2e"
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=PAIN_LABELS, yticklabels=PAIN_LABELS,
                    linewidths=0.5, linecolor="#2d2b55", cbar=False, ax=ax,
                    annot_kws={"size": 16, "weight": "bold", "color": text_color})
        ax.set_xlabel("Predicted Pain", color="white", fontsize=12)
        ax.set_ylabel("Actual Pain", color="white", fontsize=12)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold")
        ax.tick_params(colors="white")

    plt.tight_layout()
    cm_path = os.path.join(export_dir, "ensemble_overall_confusion_matrix.png")
    cm_fig.savefig(cm_path, bbox_inches="tight")
    plt.close(cm_fig)
    
    print(f"✅ Hoàn tất! Đã lưu Overall Confusion Matrix tại:\n📁 {cm_path}")

if __name__ == '__main__':
    main()