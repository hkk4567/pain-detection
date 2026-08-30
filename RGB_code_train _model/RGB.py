# -- coding: utf-8 --
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix as _cm_fn
import seaborn as _sns
import logging
from datetime import datetime
import timm
import torchvision.transforms as transforms
from collections import Counter
import matplotlib.pyplot as plt
import cv2
import re # [SỬA LỖI LEAKAGE] Thêm thư viện re để xử lý chuỗi regex

# ================================================================
# SMART FACE CLAHE — chỉ xử lý kênh L (phát sáng) trong LAB space.
# Kênh a/b (màu sắc) được giữ nguyên → ảnh đầu ra vẫn là RGB đầy đủ.
# ================================================================
class SmartFaceCLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            return img
        img_np = np.array(img.convert("RGB"))
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        lab_eq = cv2.merge((self.clahe.apply(l), a, b))   # chỉ cân bằng L
        return Image.fromarray(cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB))


# ================================================================
# PHÂN LOẠI ĐAU & HẰNG SỐ
# ================================================================
def categorize_pain(score):
    """
    Phân loại đau chuẩn lâm sàng cho DFD dataset.
    [0.0 - 2.0): Low  |  [2.0 - 4.0): Medium  |  [4.0 - MAX]: High
    """
    if score < 2.0:  return 0
    elif score < 4.0: return 1
    return 2

PAIN_LABELS = ["Low", "Medium", "High"]


# ================================================================
# CẤU HÌNH V11 — RGB + Fairness
# ================================================================
class Config:
    DATA_ROOT_DIR   = "dataset_osfstorage-archive/Stimuli"
    LABELS_PATH     = "dataset_osfstorage-archive/NormingData/DFD_NormingData.csv"
    # Dùng os.path.join — tránh hardcode backslash và đưng dẫn tuyệt đối
    SKIN_COLOR_BASE = os.path.join("dataset_osfstorage-archive", "Color Skin")

    BATCH_SIZE  = 16
    IMG_SIZE    = 300
    LEARNING_RATE = 5.0e-4
    EPOCHS_KFOLD  = 50
    K_FOLDS       = 10
    TEST_SPLIT    = 0.15

    EARLY_STOPPING_PATIENCE = 15
    CONFIDENCE_THRESHOLD    = 0.8   # OpenFace landmark quality (Baltrušaitis et al., 2018)
    BELIEVABILITY_THRESHOLD = 2.5   # Lọc ảnh diễn giả; giữ HowBelievable >= 2.5 / 5
    BACKBONE      = 'efficientnet_b3'
    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED          = 42
    WARMUP_EPOCHS = 5
    RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_V11_RGB_Fairness"
    NUM_WORKERS   = 4

cfg = Config()


# ================================================================
# HÀM BỔ TRỢ
# ================================================================
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_weights(dataset_list, logger):
    """
    Tính class weight theo group (race × gender) cho weighted loss.
    Đây là cơ chế Fairness chính: mỗi sample trong batch được nhân với
    weight của group để bù đắp mất cân bằng nhân khẩu học.
    """
    groups  = [item['group'] for item in dataset_list]
    count   = Counter(groups)
    total   = len(groups)
    n_cls   = len(count)
    weights = {}
    logger.info("\n--- WEIGHT cho từng NHÓM (WEIGHTED LOSS — Fairness) ---")
    for group, freq in sorted(count.items()):
        w = total / (n_cls * freq)
        weights[group] = w
        logger.info(f"  Nhóm {group:<20}: {freq:<4} mẫu -> Weight: {w:.4f}")
    return weights


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    imgs   = torch.stack([x[0] for x in batch])
    labels = torch.stack([x[1] for x in batch])
    groups = [x[2] for x in batch]
    return imgs, labels, groups


def save_split_filenames(indices, dataset, split_name, fold_path):
    os.makedirs(fold_path, exist_ok=True)
    txt_path = os.path.join(fold_path, f"{split_name}_list.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for idx in indices:
            item = dataset.data_list[idx]
            f.write(f"{os.path.basename(item['path'])} | {item['label']:.4f} | {item['group']}\n")


# ================================================================
# DATASET
# ================================================================
class PainDataset(Dataset):
    """
    Chỉ load metadata và PIL Image thô.
    Transform được áp bởi SplitTransformDataset để tách biệt rõ ràng
    giữa train augmentation và eval preprocessing.
    """
    def __init__(self, config):
        self.cfg = config
        self.image_path_map = self._create_smart_file_map(self.cfg.DATA_ROOT_DIR)
        self.skin_color_map = self._create_skin_color_map(self.cfg.SKIN_COLOR_BASE)

        df = pd.read_csv(self.cfg.LABELS_PATH)
        df.columns = df.columns.str.strip()
        num_cols = ['Pain_Expression', 'PhysicalPain_Neutral', 'HowBelievable',
                    'OpenFace_confidence', 'White', 'Black', 'Hispanic',
                    'EastAsian', 'SouthAsian', 'Male', 'Female']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

        if 'OpenFace_confidence' in df.columns:
            df = df[df['OpenFace_confidence'] >= self.cfg.CONFIDENCE_THRESHOLD]
        if 'HowBelievable' in df.columns:
            df = df[df['HowBelievable'].fillna(9.0) >= self.cfg.BELIEVABILITY_THRESHOLD]

        self.data_list       = []
        self.stratify_labels = []
        self.groups          = [] # [SỬA LỖI LEAKAGE] Mảng lưu Subject ID

        # [FIX HIGH #4] Theo dõi nguồn label để log minh bạch
        n_from_pain_expr    = 0
        n_from_phys_neutral = 0
        n_dropped_nan_label = 0

        for _, row in df.iterrows():
            target     = str(row['Target']).strip()
            base_name  = os.path.splitext(target)[0].lower()
            clean_name = base_name.replace('_cropped', '').replace('_standardized', '').replace('_unedited', '')

            # [SỬA LỖI LEAKAGE] Trích xuất Subject ID từ cấu trúc DPD_1_AF6_...
            parts = target.split('_')
            if len(parts) >= 3:
                subject_id = parts[2].upper() # Ví dụ: AF6
            else:
                subject_id = clean_name # Fallback nếu tên file không đúng định dạng chuẩn

            # Thử cả key gốc và biến thể _earring
            actual_key = None
            if clean_name in self.image_path_map:
                actual_key = clean_name
            elif (clean_name + "_earring") in self.image_path_map:
                actual_key = clean_name + "_earring"

            if actual_key is None:                       continue
            if actual_key not in self.skin_color_map:    continue

            full_path = self.image_path_map[actual_key]

            # [FIX HIGH #4] Pain_Expression = 0.0 hợp lệ (không đau) → dùng pd.isna().
            p_val = row['Pain_Expression']
            n_val = row['PhysicalPain_Neutral']
            if not pd.isna(p_val):
                label = float(p_val)
                n_from_pain_expr += 1
            elif not pd.isna(n_val):
                label = float(n_val)
                n_from_phys_neutral += 1
            else:
                n_dropped_nan_label += 1
                continue  # drop thay vì gán 0.0 tuỳ tiện

            race = self.skin_color_map[actual_key]   # đã đảm bảo key tồn tại ở trên

            gender   = "Male" if row.get('Male') == 1 else "Female"
            group_id = f"{race}_{gender}"

            self.data_list.append({'path': full_path, 'label': label, 'group': group_id, 'race': race})
            self.stratify_labels.append(group_id)
            self.groups.append(subject_id) # [SỬA LỖI LEAKAGE] Lưu vào mảng groups

        # Log nguồn label để đảm bảo tính minh bạch khoa học
        logging.info(f"[Label source] Pain_Expression: {n_from_pain_expr} | "
                     f"PhysicalPain_Neutral (fallback): {n_from_phys_neutral} | "
                     f"Dropped (cả 2 NaN): {n_dropped_nan_label}")

    def _create_smart_file_map(self, root_dir):
        file_map = {}
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    base_name  = os.path.splitext(f)[0].lower()
                    clean_name = base_name.replace('_cropped', '').replace('_standardized', '').replace('_unedited', '')
                    if "cropped" in dirpath.lower() or clean_name not in file_map:
                        file_map[clean_name] = os.path.join(dirpath, f)
        return file_map

    def _create_skin_color_map(self, skin_base):
        color_map = {}
        if not os.path.exists(skin_base):
            return color_map
        mapping = {'black': 'Black', 'white': 'White', 'yellow': 'Asian'}
        for folder, label in mapping.items():
            folder_path = os.path.join(skin_base, folder)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        base_name  = os.path.splitext(f)[0].lower()
                        clean_name = base_name.replace('_cropped', '').replace('_standardized', '').replace('_unedited', '')
                        color_map[clean_name] = label
        return color_map

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """Trả PIL Image thô; transform do SplitTransformDataset đảm nhận."""
        item = self.data_list[idx]
        try:
            img = ImageOps.exif_transpose(Image.open(item['path'])).convert("RGB")
            return img, torch.tensor(item['label'], dtype=torch.float), item['group']
        except Exception as e:
            logging.warning(f"[DataError] {item['path']} | {e}")
            return None


class SplitTransformDataset(Dataset):
    """
    Áp transform theo split (train / eval) lên PainDataset.
    Đây là nơi DUY NHẤT transform được áp dụng — tránh nhầm lẫn với
    mô hình cũ khi transform được truyền trực tiếp vào PainDataset.
    """
    def __init__(self, base_dataset, transform):
        self.base      = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base.data_list[idx]
        try:
            img = ImageOps.exif_transpose(Image.open(item['path'])).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(item['label'], dtype=torch.float), item['group']
        except Exception as e:
            logging.warning(f"[DataError] {item['path']} | {e}")
            return None


# ================================================================
# MODEL
# ================================================================
class BaselinePainModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone  = timm.create_model(cfg.BACKBONE, pretrained=True, num_classes=0, global_pool='avg')
        self.pain_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.backbone.num_features, 1)
        )

    def forward(self, x):
        return self.pain_head(self.backbone(x))


# ================================================================
# WARMUP + COSINE ANNEALING SCHEDULER
# ================================================================
class WarmupCosineAnnealingLR:
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = optimizer.defaults['lr']
        self.epoch         = 0

    def step(self):
        if self.epoch < self.warmup_epochs:
            lr = self.base_lr * (self.epoch + 1) / self.warmup_epochs
        else:
            progress = (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * (1 + np.cos(np.pi * progress)) / 2
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.epoch += 1


# ================================================================
# MAIN
# ================================================================
def main(config):
    seed_everything(config.SEED)

    # --- Logging: console + file ---
    export_base   = os.path.join("exported_data", config.RUN_TIMESTAMP)
    os.makedirs(export_base, exist_ok=True)
    log_file_path = os.path.join(export_base, "training_log.txt")

    logger = logging.getLogger('root')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter('%(message)s')
    for h in [logging.StreamHandler(),
              logging.FileHandler(log_file_path, mode='w', encoding='utf-8')]:
        h.setFormatter(fmt)
        logger.addHandler(h)

    # --- CSV metric logs (thay wandb) ---
    epoch_log_path = os.path.join(export_base, "epoch_metrics.csv")
    fold_log_path  = os.path.join(export_base, "fold_metrics.csv")
    test_log_path  = os.path.join(export_base, "ensemble_test_metrics.csv")

    with open(epoch_log_path, "w", encoding="utf-8") as f:
        f.write("fold,epoch,train_loss,val_loss,lr\n")
    with open(fold_log_path, "w", encoding="utf-8") as f:
        f.write("fold,val_r2,val_rmse,val_mae\n")
    with open(test_log_path, "w", encoding="utf-8") as f:
        f.write("ensemble_test_r2,ensemble_test_rmse,ensemble_test_mae\n")

    # --- Transforms ---
    clahe = SmartFaceCLAHE(clip_limit=2.0, tile_grid_size=(8, 8))

    RGB_MEAN = [0.485, 0.456, 0.406]
    RGB_STD  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        clahe,                                                   
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
    ])
    eval_transform = transforms.Compose([
        clahe,
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
    ])

    # --- Dataset ---
    full_dataset  = PainDataset(config)
    train_dataset = SplitTransformDataset(full_dataset, train_transform)
    eval_dataset  = SplitTransformDataset(full_dataset, eval_transform)

    if len(full_dataset) == 0:
        logger.info("❌ Không có dữ liệu để huấn luyện!")
        return

    logger.info("\n" + "="*60)
    logger.info("🚀 BẮT ĐẦU HUẤN LUYỆN MODEL V11 (RGB + FAIRNESS + CLAHE)")
    logger.info("="*60)
    logger.info(f"🖥️  Thiết bị   : {str(config.DEVICE).upper()}")
    logger.info(f"📂  Run        : {config.RUN_TIMESTAMP}")
    logger.info(f"📊  Cấu hình   : {config.K_FOLDS}-Folds | Test: {config.TEST_SPLIT*100:.0f}%")
    logger.info(f"📁  Skin map   : {len(full_dataset.skin_color_map)} ảnh (Black/White/Asian)")
    logger.info(f"📈  Tổng mẫu   : {len(full_dataset)} (sau lọc confidence & believability)")
    logger.info(f"👤  Số Subjects: {len(set(full_dataset.groups))}") # [SỬA LỖI LEAKAGE] Log tổng số Subject
    logger.info("="*60)

    # --- Chia hold-out test ---
    all_indices = np.arange(len(full_dataset))
    all_labels  = np.array(full_dataset.stratify_labels)
    all_groups  = np.array(full_dataset.groups) # [SỬA LỖI LEAKAGE] Lấy danh sách groups (Subject ID)

    counts      = Counter(all_labels)
    rare_groups = [g for g, c in counts.items() if c < 2]
    if rare_groups:
        logger.info(f"⚠️  Nhóm {rare_groups} chỉ có 1 mẫu → loại khỏi Stratify.")
        mask        = np.isin(all_labels, rare_groups, invert=True)
        all_indices = all_indices[mask]
        all_labels  = all_labels[mask]
        all_groups  = all_groups[mask] # [SỬA LỖI LEAKAGE] Cập nhật mảng groups sau khi mask

    # =========================================================================
    # [CUSTOM SPLIT] CHIA TẬP TEST THEO TỶ LỆ CỐ ĐỊNH 1:2:3 (ASIAN:BLACK:WHITE)
    # Bằng cách lấy theo Subject ID để tránh Data Leakage
    # =========================================================================
    # 1. Gom nhóm Subject theo Race
    subject_to_race = {}
    for idx in all_indices:
        subj = full_dataset.groups[idx]
        race = full_dataset.data_list[idx]['race']
        subject_to_race[subj] = race

    asian_subjects = [s for s, r in subject_to_race.items() if r == 'Asian']
    black_subjects = [s for s, r in subject_to_race.items() if r == 'Black']
    white_subjects = [s for s, r in subject_to_race.items() if r == 'White']

    # 2. Xáo trộn danh sách Subject với Seed cố định
    import random
    rng = random.Random(config.SEED)
    rng.shuffle(asian_subjects)
    rng.shuffle(black_subjects)
    rng.shuffle(white_subjects)

    # 3. Chọn số lượng Subject cho tập Test (Tỷ lệ 1:2:3)
    # Tổng Subject = 226. Chọn 6 Asian, 12 Black, 18 White (Tổng: 36 subjects ~ 15.9%)
    n_asian = 6
    n_black = 12
    n_white = 18

    test_subjects = set(
        asian_subjects[:n_asian] + 
        black_subjects[:n_black] + 
        white_subjects[:n_white]
    )

    # 4. Phân bổ index dựa trên việc Subject có thuộc tập Test hay không
    train_val_idx_list = []
    test_idx_list = []
    for idx in all_indices:
        if full_dataset.groups[idx] in test_subjects:
            test_idx_list.append(idx)
        else:
            train_val_idx_list.append(idx)

    train_val_idx = np.array(train_val_idx_list)
    test_idx = np.array(test_idx_list)
    # =========================================================================

    # test_loader dùng eval_dataset (có eval_transform)
    test_loader = DataLoader(
        Subset(eval_dataset, test_idx),
        batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    # --- Log phân bố màu da ---
    def get_dist(indices):
        races = [full_dataset.data_list[i]['race'] for i in indices]
        c     = Counter(races)
        total = len(indices)
        return {r: (n, n / total * 100) for r, n in c.items()}

    tv_dist, te_dist = get_dist(train_val_idx), get_dist(test_idx)
    all_races        = sorted(set(list(tv_dist) + list(te_dist)))

    logger.info("\n🔍 PHÂN BỔ MÀU DA:")
    logger.info(f"{'Nhóm':<15} | {'Train+Val':>12} | {'Train %':>8} | {'Test':>6} | {'Test %':>7}")
    logger.info("-" * 60)
    skin_log_path = os.path.join(export_base, "skin_color_distribution.txt")
    with open(skin_log_path, 'w', encoding='utf-8') as f:
        f.write(f"--- MÀU DA (RUN: {config.RUN_TIMESTAMP}) ---\n\n")
        for r in all_races:
            tr_c, tr_p = tv_dist.get(r, (0, 0.0))
            te_c, te_p = te_dist.get(r, (0, 0.0))
            line = f"{r:<15} | {tr_c:>12} | {tr_p:>7.2f}% | {te_c:>6} | {te_p:>7.2f}%"
            logger.info(line)
            f.write(line + "\n")
    logger.info("-" * 60)
    logger.info(f"✅ Đã lưu: {skin_log_path}\n")

    # --- K-Fold ---
    # [SỬA LỖI LEAKAGE] Dùng StratifiedGroupKFold thay vì StratifiedKFold
    sgkf = StratifiedGroupKFold(n_splits=config.K_FOLDS, shuffle=True, random_state=config.SEED)
    y_train_val = [full_dataset.stratify_labels[i] for i in train_val_idx]
    groups_train_val = [full_dataset.groups[i] for i in train_val_idx] # [SỬA LỖI LEAKAGE] Truyền subject id cho K-Fold
    
    fold_results = []

    # [SỬA LỖI LEAKAGE] Truyền groups=groups_train_val vào split
    for fold, (tr_rel, val_rel) in enumerate(sgkf.split(train_val_idx, y_train_val, groups=groups_train_val)):
        logger.info(f"\n{'='*60}")
        logger.info(f"  FOLD {fold+1}/{config.K_FOLDS}")
        logger.info(f"{'='*60}")

        fold_export_path = os.path.join(export_base, f"fold_{fold+1}")
        train_ids = train_val_idx[tr_rel]
        val_ids   = train_val_idx[val_rel]

        save_split_filenames(train_ids, full_dataset, "train", fold_export_path)
        save_split_filenames(val_ids,   full_dataset, "val",   fold_export_path)
        save_split_filenames(test_idx,  full_dataset, "test",  fold_export_path)

        # [FIX CRITICAL #1] Tính group_weights CHỈ trên train set của fold này
        fold_train_data_list = [full_dataset.data_list[i] for i in train_ids]
        group_weights = calculate_weights(fold_train_data_list, logger)

        train_loader = DataLoader(
            Subset(train_dataset, train_ids),
            batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate,
            num_workers=config.NUM_WORKERS, pin_memory=True
        )
        val_loader = DataLoader(
            Subset(eval_dataset, val_ids),
            batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate,
            num_workers=config.NUM_WORKERS, pin_memory=True
        )

        model     = BaselinePainModel(config).to(config.DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = WarmupCosineAnnealingLR(optimizer,
                                            warmup_epochs=config.WARMUP_EPOCHS,
                                            total_epochs=config.EPOCHS_KFOLD)
        scheduler.step()

        best_val_loss    = float('inf')
        patience_counter = 0

        checkpoint_path = os.path.join("checkpoints", f"{config.RUN_TIMESTAMP}_fold_{fold+1}.pth")
        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(config.EPOCHS_KFOLD):
            # --- TRAIN ---
            model.train()
            train_loss_sum = 0.0
            train_n_samples = 0
            for batch_data in train_loader:
                if batch_data is None:
                    continue
                imgs, labels, batch_groups = batch_data
                imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)

                preds = model(imgs).squeeze(-1)
                batch_w = torch.tensor(
                    [group_weights[g] for g in batch_groups],
                    dtype=preds.dtype, device=config.DEVICE
                )
                loss  = ((preds - labels) ** 2 * batch_w).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * labels.size(0)
                train_n_samples += labels.size(0)

            # --- VALIDATION ---
            model.eval()
            val_loss_sum = 0.0
            val_n_samples = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    if batch_data is None:
                        continue
                    imgs, labels, batch_groups = batch_data
                    imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
                    preds = model(imgs).squeeze(-1)
                    
                    batch_w = torch.tensor(
                        [group_weights[g] for g in batch_groups],
                        dtype=preds.dtype, device=config.DEVICE
                    )
                    
                    batch_loss = ((preds - labels) ** 2 * batch_w).mean()
                    
                    val_loss_sum += batch_loss.item() * labels.size(0)
                    val_n_samples += labels.size(0)

            avg_train = train_loss_sum / train_n_samples if train_n_samples > 0 else 0.0
            avg_val   = val_loss_sum / val_n_samples if val_n_samples > 0 else 0.0
            cur_lr    = optimizer.param_groups[0]['lr']

            logger.info(f" Ep {epoch+1:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {cur_lr:.6f}")

            with open(epoch_log_path, "a", encoding="utf-8") as f:
                f.write(f"{fold+1},{epoch+1},{avg_train:.8f},{avg_val:.8f},{cur_lr:.10f}\n")

            if avg_val < best_val_loss:
                best_val_loss    = avg_val
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    logger.info("  ⏹️  Early stopping triggered!")
                    break

            scheduler.step()

        # --- ĐÁNH GIÁ VALIDATION (per fold) ---
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True)
        )
        model.eval()
        all_labels_v, all_preds_v = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                if batch_data is None:
                    continue
                imgs, labels, _ = batch_data
                preds = model(imgs.to(config.DEVICE)).squeeze(-1)
                all_labels_v.extend(labels.cpu().numpy())
                all_preds_v.extend(preds.cpu().numpy())

        r2   = r2_score(all_labels_v, all_preds_v)
        rmse = np.sqrt(mean_squared_error(all_labels_v, all_preds_v))
        mae  = mean_absolute_error(all_labels_v, all_preds_v)

        fold_results.append({'fold': fold + 1, 'val_r2': r2, 'val_rmse': rmse, 'val_mae': mae})
        logger.info(f"→ Fold {fold+1} Val | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

        with open(fold_log_path, "a", encoding="utf-8") as f:
            f.write(f"{fold+1},{r2:.8f},{rmse:.8f},{mae:.8f}\n")

        # Scatter plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(all_labels_v, all_preds_v, alpha=0.5, s=20, color='steelblue', label='Samples')
        _lo = min(min(all_labels_v), min(all_preds_v))
        _hi = max(max(all_labels_v), max(all_preds_v))
        ax.plot([_lo, _hi], [_lo, _hi], 'r--', lw=1.5, label='Perfect fit')
        ax.set_xlabel("Actual Pain Score")
        ax.set_ylabel("Predicted Pain Score")
        ax.set_title(f"Fold {fold+1} | R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(export_base, f"scatter_fold_{fold+1}.png"), bbox_inches="tight")
        plt.close(fig)

        # Confusion matrix
        _yt  = [categorize_pain(x) for x in all_labels_v]
        _yp  = [categorize_pain(x) for x in all_preds_v]
        _cm  = _cm_fn(_yt, _yp, labels=[0, 1, 2])
        _rs  = _cm.sum(axis=1, keepdims=True)
        _cmr = np.divide(_cm.astype(float), _rs, out=np.zeros_like(_cm, dtype=float), where=_rs != 0)
        _off = _cm.copy(); np.fill_diagonal(_off, 0)
        _mi  = np.unravel_index(_off.argmax(), _off.shape)
        _worst = (f"⚠ Nhầm: '{PAIN_LABELS[_mi[0]]}' → '{PAIN_LABELS[_mi[1]]}' ({_off[_mi]} lần)"
                  if _off[_mi] > 0 else "✓  OK")

        cm_fig, cm_axes = plt.subplots(1, 2, figsize=(13, 5))
        cm_fig.patch.set_facecolor("#1a1a2e")
        for _ax, _data, _fmt, _cmap, _sub in [
            (cm_axes[0], _cm,  "d",    _sns.color_palette(["#2d2b55","#4a4080","#9b59b6","#ff79c6"], as_cmap=True), "Count"),
            (cm_axes[1], _cmr, ".2f",  "YlGn", "Recall"),
        ]:
            _ax.set_facecolor("#16213e")
            _tc = "white" if _sub == "Count" else "#1a1a2e"
            _sns.heatmap(_data, annot=True, fmt=_fmt, cmap=_cmap,
                         xticklabels=PAIN_LABELS, yticklabels=PAIN_LABELS,
                         linewidths=0.5, linecolor="#2d2b55", cbar=False, ax=_ax,
                         annot_kws={"size": 16, "weight": "bold", "color": _tc})
            _ax.set_xlabel("Predicted", color="white", fontsize=12)
            _ax.set_ylabel("Actual",    color="white", fontsize=12)
            _ax.set_title(f"{_sub} – Fold {fold+1}", color="white", fontsize=13, fontweight="bold")
            _ax.tick_params(colors="white")
        cm_fig.text(0.5, -0.04, _worst, ha="center", color="#ff6b6b", fontsize=11)
        plt.tight_layout()
        cm_fig.savefig(os.path.join(export_base, f"cm_fold_{fold+1}.png"), bbox_inches="tight")
        plt.close(cm_fig)

    # ================================================================
    # ENSEMBLE TEST EVALUATION
    # ================================================================
    logger.info("\n" + "="*60)
    logger.info(f"ENSEMBLE TEST ({config.K_FOLDS} fold checkpoints)")
    logger.info("="*60)

    ensemble_models = []
    for i in range(config.K_FOLDS):
        cp = os.path.join("checkpoints", f"{config.RUN_TIMESTAMP}_fold_{i+1}.pth")
        if os.path.exists(cp):
            m = BaselinePainModel(config).to(config.DEVICE)
            m.load_state_dict(torch.load(cp, map_location=config.DEVICE, weights_only=True))
            m.eval()
            ensemble_models.append(m)
        else:
            logger.info(f"[WARN] Không tìm thấy checkpoint: {cp}")

    logger.info(f"Loaded {len(ensemble_models)}/{config.K_FOLDS} checkpoints")

    if ensemble_models:
        all_labels_te, all_preds_te = [], []
        with torch.no_grad():
            for batch_data in test_loader:
                if batch_data is None:
                    continue
                imgs, labels, _ = batch_data
                imgs = imgs.to(config.DEVICE)
                ens_pred = np.mean([m(imgs).squeeze(-1).cpu().numpy() for m in ensemble_models], axis=0)
                all_labels_te.extend(labels.numpy())
                all_preds_te.extend(ens_pred.tolist())

        te_r2   = r2_score(all_labels_te, all_preds_te)
        te_rmse = np.sqrt(mean_squared_error(all_labels_te, all_preds_te))
        te_mae  = mean_absolute_error(all_labels_te, all_preds_te)

        logger.info(f"ENSEMBLE TEST | RMSE: {te_rmse:.4f} | MAE: {te_mae:.4f} | R²: {te_r2:.4f}")

        with open(test_log_path, "a", encoding="utf-8") as f:
            f.write(f"{te_r2:.8f},{te_rmse:.8f},{te_mae:.8f}\n")

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(all_labels_te, all_preds_te, alpha=0.5, s=25, color='darkorange', label='Ensemble')
        _lo = min(min(all_labels_te), min(all_preds_te))
        _hi = max(max(all_labels_te), max(all_preds_te))
        ax.plot([_lo, _hi], [_lo, _hi], 'r--', lw=1.5, label='Perfect fit')
        ax.set_xlabel("Actual Pain Score"); ax.set_ylabel("Predicted Pain Score")
        ax.set_title(f"ENSEMBLE TEST | R²={te_r2:.3f}  RMSE={te_rmse:.3f}  MAE={te_mae:.3f}")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        ens_path = os.path.join(export_base, "scatter_ensemble_test.png")
        fig.savefig(ens_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"✅ Đã lưu: {ens_path}")

    # ================================================================
    # BẢNG TỔNG KẾT
    # ================================================================
    results_df = pd.DataFrame(fold_results)
    if not results_df.empty:
        mean_r2   = results_df['val_r2'].mean()
        std_r2    = results_df['val_r2'].std()
        mean_rmse = results_df['val_rmse'].mean()
        std_rmse  = results_df['val_rmse'].std()
        mean_mae  = results_df['val_mae'].mean()
        std_mae   = results_df['val_mae'].std()
        best_fold = results_df.loc[results_df['val_r2'].idxmax()]

        logger.info("\n" + "="*60)
        logger.info(f"--- V11 RGB FINAL RESULTS ({config.K_FOLDS}-Fold Stratified) ---")
        logger.info("="*60)
        logger.info(f"{'fold':>6} {'r2':>12} {'rmse':>12} {'mae':>12}")
        for _, row in results_df.iterrows():
            logger.info(f"{int(row['fold']):>6} {row['val_r2']:>12.6f} {row['val_rmse']:>12.6f} {row['val_mae']:>12.6f}")

        logger.info(f"\nMean RMSE : {mean_rmse:.4f} (±{std_rmse:.4f})")
        logger.info(f"Mean MAE  : {mean_mae:.4f}  (±{std_mae:.4f})")
        logger.info(f"Mean R²   : {mean_r2:.4f}  (±{std_r2:.4f})")
        logger.info(f"🏆 Best   : Fold {int(best_fold['fold'])} (R²={best_fold['val_r2']:.4f}, RMSE={best_fold['val_rmse']:.4f})")

        r2_check   = "✓" if mean_r2   >= 0.68 else "✗"
        rmse_check = "✓" if mean_rmse <= 0.62 else "✗"
        std_check  = "✓" if std_r2    <= 0.04 else "✗"
        logger.info(f"\n{r2_check}  R²   target ≥0.68 | đạt: {mean_r2:.4f}")
        logger.info(f"{rmse_check}  RMSE target ≤0.62 | đạt: {mean_rmse:.4f}")
        logger.info(f"{std_check}  Std  target ≤4%   | đạt: ±{std_r2*100:.2f}%")
        logger.info("="*60)
        logger.info(f"✅ HOÀN TẤT! Log: {log_file_path}")
        logger.info(f"✅ Epoch CSV: {epoch_log_path}")
        logger.info(f"✅ Fold  CSV: {fold_log_path}")
        logger.info(f"✅ Test  CSV: {test_log_path}")


if __name__ == '__main__':
    if sys.platform == "win32":
        cfg.NUM_WORKERS = 0
    main(cfg)