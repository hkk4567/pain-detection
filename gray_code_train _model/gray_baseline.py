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
import re # [SỬA LỖI LEAKAGE] Thêm thư viện re để xử lý chuỗi

# ================================================================
# SMART FACE CLAHE
# ================================================================
class SmartFaceCLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            return img
        img_np = np.array(img.convert("RGB"))
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_adjusted = self.clahe.apply(l)
        lab_adjusted = cv2.merge((l_adjusted, a, b))
        img_normalized = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_normalized)


def categorize_pain(score):
    if score < 2.0:
        return 0
    elif score < 4.0:
        return 1
    return 2


PAIN_LABELS = ["Low", "Medium", "High"]


# === CẤU HÌNH V11 ===
class Config:
    DATA_ROOT_DIR = "dataset_osfstorage-archive/Stimuli"
    LABELS_PATH = "dataset_osfstorage-archive/NormingData/DFD_NormingData.csv"
    SKIN_COLOR_BASE = os.path.join("dataset_osfstorage-archive", "Color Skin")

    BATCH_SIZE = 16
    IMG_SIZE = 300
    LEARNING_RATE = 5.0e-4
    EPOCHS_KFOLD = 50
    K_FOLDS = 10
    TEST_SPLIT = 0.15

    EARLY_STOPPING_PATIENCE = 15
    CONFIDENCE_THRESHOLD = 0.8
    BELIEVABILITY_THRESHOLD = 2.5
    BACKBONE = 'efficientnet_b3'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    WARMUP_EPOCHS = 5
    RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_GRAY_PureRaw_Baseline"
    NUM_WORKERS = 4


cfg = Config()


# === HÀM BỔ TRỢ ===
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    imgs = torch.stack([x[0] for x in batch])
    labels = torch.stack([x[1] for x in batch])
    groups = [x[2] for x in batch]
    return imgs, labels, groups


def calculate_weights(dataset_list, logger):
    """
    Tính class weight theo group (race × gender) cho weighted loss.
    """
    groups = [item['group'] for item in dataset_list]
    count = Counter(groups)
    total = len(groups)
    n_cls = len(count)
    weights = {}

    logger.info("\n--- WEIGHT cho từng NHÓM (WEIGHTED LOSS — Fairness) ---")
    for group, freq in sorted(count.items()):
        w = total / (n_cls * freq)
        weights[group] = w
        logger.info(f"  Nhóm {group:<20}: {freq:<4} mẫu -> Weight: {w:.4f}")
    return weights


def save_split_filenames(indices, dataset, split_name, fold_path):
    os.makedirs(fold_path, exist_ok=True)
    txt_path = os.path.join(fold_path, f"{split_name}_list.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for idx in indices:
            item = dataset.data_list[idx]
            f.write(f"{os.path.basename(item['path'])} | {item['label']:.4f} | {item['group']}\n")


# === DATASET ===
class PainDataset(Dataset):
    def __init__(self, config):
        self.cfg = config
        self.image_path_map = self._create_smart_file_map(self.cfg.DATA_ROOT_DIR)
        self.skin_color_map = self._create_skin_color_map(self.cfg.SKIN_COLOR_BASE)

        df = pd.read_csv(self.cfg.LABELS_PATH)
        df.columns = df.columns.str.strip()
        cols = ['Pain_Expression', 'PhysicalPain_Neutral', 'HowBelievable', 'OpenFace_confidence',
                'White', 'Black', 'Hispanic', 'EastAsian', 'SouthAsian', 'Male', 'Female']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

        if 'OpenFace_confidence' in df.columns:
            df = df[df['OpenFace_confidence'] >= self.cfg.CONFIDENCE_THRESHOLD]
        if 'HowBelievable' in df.columns:
            df = df[df['HowBelievable'].fillna(9.0) >= self.cfg.BELIEVABILITY_THRESHOLD]

        self.data_list = []
        self.stratify_labels = []
        self.groups = [] # [SỬA LỖI LEAKAGE] Mảng lưu Subject ID

        # [FIX HIGH #4] Theo dõi nguồn label để log minh bạch
        n_from_pain_expr    = 0
        n_from_phys_neutral = 0
        n_dropped_nan_label = 0

        for _, row in df.iterrows():
            target = str(row['Target']).strip()
            base_name = os.path.splitext(target)[0].lower()
            clean_name = base_name.replace('_cropped', '').replace('_standardized', '').replace('_unedited', '')

            # [SỬA LỖI LEAKAGE] Trích xuất Subject ID từ cấu trúc DPD_1_AF6_...
            parts = target.split('_')
            if len(parts) >= 3:
                subject_id = parts[2].upper() # Ví dụ: AF6
            else:
                subject_id = clean_name # Fallback nếu tên file không đúng định dạng chuẩn

            actual_key = None
            if clean_name in self.image_path_map:
                actual_key = clean_name
            elif (clean_name + "_earring") in self.image_path_map:
                actual_key = clean_name + "_earring"

            if actual_key is None:
                continue
            if actual_key not in self.skin_color_map:
                continue

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

            race = self.skin_color_map[actual_key]
            gender = "Male" if row.get('Male') == 1 else "Female"
            group_id = f"{race}_{gender}"

            self.data_list.append({'path': full_path, 'label': label, 'group': group_id, 'race': race})
            self.stratify_labels.append(group_id)
            self.groups.append(subject_id) # [SỬA LỖI LEAKAGE] Lưu vào mảng groups

        logging.info(f"[Label source] Pain_Expression: {n_from_pain_expr} | "
                     f"PhysicalPain_Neutral (fallback): {n_from_phys_neutral} | "
                     f"Dropped (cả 2 NaN): {n_dropped_nan_label}")

    def _create_smart_file_map(self, root_dir):
        file_map = {}
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    base_name = os.path.splitext(f)[0].lower()
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
                        base_name = os.path.splitext(f)[0].lower()
                        clean_name = base_name.replace('_cropped', '').replace('_standardized', '').replace('_unedited', '')
                        color_map[clean_name] = label
        return color_map

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        try:
            img = ImageOps.exif_transpose(Image.open(item['path'])).convert("RGB")
            return img, torch.tensor(item['label'], dtype=torch.float), item['group']
        except Exception as e:
            logging.warning(f"[DataError] {item['path']} | {e}")
            return None


class SplitTransformDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
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


# === MODEL ===
class BaselinePainModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(cfg.BACKBONE, pretrained=True, num_classes=0, global_pool='avg')
        self.pain_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.backbone.num_features, 1)
        )

    def forward(self, x):
        return self.pain_head(self.backbone(x))


# === WARMUP + COSINE SCHEDULER ===
class WarmupCosineAnnealingLR:
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = optimizer.defaults['lr']
        self.epoch = 0

    def step(self):
        if self.epoch < self.warmup_epochs:
            lr = self.base_lr * (self.epoch + 1) / self.warmup_epochs
        else:
            progress = (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * (1 + np.cos(np.pi * progress)) / 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.epoch += 1


# === MAIN TRAINING FUNCTION ===
def main(config):
    seed_everything(config.SEED)

    export_base = os.path.join("exported_data", config.RUN_TIMESTAMP)
    os.makedirs(export_base, exist_ok=True)
    log_file_path = os.path.join(export_base, "training_log.txt")

    # --- CSV metric logs ---
    epoch_log_path = os.path.join(export_base, "epoch_metrics.csv")
    fold_log_path  = os.path.join(export_base, "fold_metrics.csv")
    test_log_path  = os.path.join(export_base, "ensemble_test_metrics.csv")

    with open(epoch_log_path, "w", encoding="utf-8") as f:
        f.write("fold,epoch,train_loss,val_loss,lr\n")
    with open(fold_log_path, "w", encoding="utf-8") as f:
        f.write("fold,val_r2,val_rmse,val_mae\n")
    with open(test_log_path, "w", encoding="utf-8") as f:
        f.write("ensemble_test_r2,ensemble_test_rmse,ensemble_test_mae\n")

    logger = logging.getLogger('root')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    clahe = SmartFaceCLAHE(clip_limit=2.0, tile_grid_size=(8, 8))

    train_transform = transforms.Compose([
        clahe,
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.449, 0.449, 0.449], std=[0.226, 0.226, 0.226])
    ])
    eval_transform = transforms.Compose([
        clahe,
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.449, 0.449, 0.449], std=[0.226, 0.226, 0.226])
    ])

    full_dataset = PainDataset(config)
    train_dataset = SplitTransformDataset(full_dataset, train_transform)
    eval_dataset = SplitTransformDataset(full_dataset, eval_transform)

    if len(full_dataset) == 0:
        logger.info("❌ Không có dữ liệu để huấn luyện!")
        return

    logger.info("\n" + "=" * 50)
    logger.info("🚀 BẮT ĐẦU HUẤN LUYỆN MODEL V11 (GRAY BASELINE | FAIRNESS-ALIGNED)")
    logger.info("=" * 50)
    logger.info(f"🖥️ Thiết bị sử dụng: {str(config.DEVICE).upper()}")
    logger.info(f"📂 Thư mục chạy: {config.RUN_TIMESTAMP}")
    logger.info(f"📊 Cấu hình: {config.K_FOLDS}-Folds | Test size: {config.TEST_SPLIT*100}%")
    logger.info(f"📁 Tổng số ảnh skin map (Black/White/Asian): {len(full_dataset.skin_color_map)}")
    logger.info(f"📈 Tổng cộng: {len(full_dataset)} ảnh phù hợp tiêu chí lọc.")
    logger.info(f"👤 Số Subjects: {len(set(full_dataset.groups))}") # [SỬA LỖI LEAKAGE]
    logger.info("=" * 50)

    all_indices = np.arange(len(full_dataset))
    all_labels = np.array(full_dataset.stratify_labels)
    all_groups = np.array(full_dataset.groups) # [SỬA LỖI LEAKAGE]

    counts = Counter(all_labels)
    rare_groups = [g for g, c in counts.items() if c < 2]
    if rare_groups:
        logger.info(f"⚠️ Cảnh báo: Các nhóm {rare_groups} chỉ có 1 mẫu. Sẽ loại bỏ để Stratify.")
        valid_mask = np.isin(all_labels, rare_groups, invert=True)
        all_indices = all_indices[valid_mask]
        all_labels = all_labels[valid_mask]
        all_groups = all_groups[valid_mask] # [SỬA LỖI LEAKAGE]

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

    test_loader = DataLoader(
        Subset(eval_dataset, test_idx),
        batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    log_path = os.path.join(export_base, "skin_color_distribution.txt")

    def get_dist(indices):
        races = [full_dataset.data_list[i]['race'] for i in indices]
        count = Counter(races)
        total = len(indices)
        return {r: (c, c / total * 100) for r, c in count.items()}

    train_val_dist, test_dist = get_dist(train_val_idx), get_dist(test_idx)

    logger.info("\n🔍 PHÂN BỔ MÀU DA TRONG DỮ LIỆU:")
    logger.info(f"{'Nhóm':<15} | {'Train+Val (Count/%)':<25} | {'Test (Count/%)':<25}")
    logger.info("-" * 75)
    all_races = sorted(list(set(list(train_val_dist.keys()) + list(test_dist.keys()))))
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"--- THỐNG KÊ TỶ LỆ MÀU DA (RUN: {config.RUN_TIMESTAMP}) ---\n\n")
        header = f"{'Nhóm':<15} | {'Train+Val Count':<15} | {'Train %':<10} | {'Test Count':<12} | {'Test %':<10}\n"
        f.write(header + "-" * 75 + "\n")
        for r in all_races:
            tr_c, tr_p = train_val_dist.get(r, (0, 0))
            te_c, te_p = test_dist.get(r, (0, 0))
            line = f"{r:<15} | {tr_c:<15} | {tr_p:>7.2f}% | {te_c:<12} | {te_p:>7.2f}%"
            logger.info(line)
            f.write(line + "\n")
    logger.info("-" * 75)
    logger.info(f"✅ Đã lưu log chi tiết vào: {log_path}\n")

    # [SỬA LỖI LEAKAGE] Dùng StratifiedGroupKFold thay vì StratifiedKFold
    sgkf = StratifiedGroupKFold(n_splits=config.K_FOLDS, shuffle=True, random_state=config.SEED)
    y_train_val = [full_dataset.stratify_labels[i] for i in train_val_idx]
    groups_train_val = [full_dataset.groups[i] for i in train_val_idx] # [SỬA LỖI LEAKAGE]
    
    fold_results = []

    # [SỬA LỖI LEAKAGE] Truyền groups=groups_train_val vào sgkf.split
    for fold, (train_fold_rel_idx, val_fold_rel_idx) in enumerate(sgkf.split(train_val_idx, y_train_val, groups=groups_train_val)):
        logger.info(f"\n=== FOLD {fold+1}/{config.K_FOLDS} ===")
        fold_export_path = os.path.join(export_base, f"fold_{fold+1}")

        train_ids = train_val_idx[train_fold_rel_idx]
        val_ids = train_val_idx[val_fold_rel_idx]

        save_split_filenames(train_ids, full_dataset, "train", fold_export_path)
        save_split_filenames(val_ids, full_dataset, "val", fold_export_path)
        save_split_filenames(test_idx, full_dataset, "test", fold_export_path)

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

        model = BaselinePainModel(config).to(config.DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = WarmupCosineAnnealingLR(
            optimizer, warmup_epochs=config.WARMUP_EPOCHS, total_epochs=config.EPOCHS_KFOLD
        )
        scheduler.step()

        best_val_loss = float('inf')
        patience_counter = 0

        checkpoint_path = os.path.join("checkpoints", f"{config.RUN_TIMESTAMP}_fold_{fold+1}.pth")
        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(config.EPOCHS_KFOLD):
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
                loss = ((preds - labels) ** 2 * batch_w).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * labels.size(0)
                train_n_samples += labels.size(0)

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

            avg_train_loss = train_loss_sum / train_n_samples if train_n_samples > 0 else 0.0
            avg_val_loss = val_loss_sum / val_n_samples if val_n_samples > 0 else 0.0
            current_lr = optimizer.param_groups[0]['lr']

            logger.info(f" Ep {epoch+1:02d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

            with open(epoch_log_path, "a", encoding="utf-8") as f:
                f.write(f"{fold+1},{epoch+1},{avg_train_loss:.8f},{avg_val_loss:.8f},{current_lr:.10f}\n")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    logger.info("Early stopping triggered!")
                    break

            scheduler.step()

        model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True))
        model.eval()
        all_labels_t, all_preds_t = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                if batch_data is None:
                    continue
                imgs, labels, _ = batch_data
                preds = model(imgs.to(config.DEVICE)).squeeze(-1)
                all_labels_t.extend(labels.cpu().numpy())
                all_preds_t.extend(preds.cpu().numpy())

        r2 = r2_score(all_labels_t, all_preds_t)
        rmse = np.sqrt(mean_squared_error(all_labels_t, all_preds_t))
        mae = mean_absolute_error(all_labels_t, all_preds_t)

        fold_results.append({'fold': fold+1, 'val_r2': r2, 'val_rmse': rmse, 'val_mae': mae})
        logger.info(f"-> Fold {fold+1} Validation | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

        with open(fold_log_path, "a", encoding="utf-8") as f:
            f.write(f"{fold+1},{r2:.8f},{rmse:.8f},{mae:.8f}\n")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(all_labels_t, all_preds_t, alpha=0.5, s=20, color='steelblue', label='Samples')
        _min = min(min(all_labels_t), min(all_preds_t))
        _max = max(max(all_labels_t), max(all_preds_t))
        ax.plot([_min, _max], [_min, _max], 'r--', linewidth=1.5, label='Perfect fit')
        ax.set_xlabel("Actual Pain Score")
        ax.set_ylabel("Predicted Pain Score")
        ax.set_title(f"Fold {fold+1} | R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        scatter_path = os.path.join(export_base, f"scatter_fold_{fold+1}.png")
        fig.savefig(scatter_path, bbox_inches="tight")
        plt.close(fig)

        _labels = PAIN_LABELS
        _yt = [categorize_pain(x) for x in all_labels_t]
        _yp = [categorize_pain(x) for x in all_preds_t]

        _cm = _cm_fn(_yt, _yp, labels=[0, 1, 2])
        _rs = _cm.sum(axis=1, keepdims=True)
        _cmr = np.divide(_cm.astype(float), _rs, out=np.zeros_like(_cm, dtype=float), where=_rs != 0)
        _off = _cm.copy()
        np.fill_diagonal(_off, 0)
        _mi = np.unravel_index(_off.argmax(), _off.shape)
        _worst = (f"⚠  Nhầm: '{_labels[_mi[0]]}' → '{_labels[_mi[1]]}' ({_off[_mi]} lần)"
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
                         xticklabels=_labels, yticklabels=_labels,
                         linewidths=0.5, linecolor="#2d2b55", cbar=False, ax=_ax,
                         annot_kws={"size": 16, "weight": "bold", "color": _tc})
            _ax.set_xlabel("Predicted", color="white", fontsize=12)
            _ax.set_ylabel("Actual",    color="white", fontsize=12)
            _ax.set_title(f"{_sub} – Fold {fold+1}", color="white", fontsize=13, fontweight="bold")
            _ax.tick_params(colors="white")
        cm_fig.text(0.5, -0.04, _worst, ha="center", color="#ff6b6b", fontsize=11)
        plt.tight_layout()
        cm_path = os.path.join(export_base, f"cm_fold_{fold+1}.png")
        cm_fig.savefig(cm_path, bbox_inches="tight")
        plt.close(cm_fig)

    logger.info("\n" + "=" * 80)
    logger.info(f"ENSEMBLE TEST EVALUATION ({config.K_FOLDS} folds -> đánh giá 1 lần duy nhất)")
    logger.info("=" * 80)

    checkpoint_paths_all = [
        os.path.join("checkpoints", f"{config.RUN_TIMESTAMP}_fold_{i+1}.pth")
        for i in range(config.K_FOLDS)
    ]

    ensemble_models = []
    for cp in checkpoint_paths_all:
        if os.path.exists(cp):
            m = BaselinePainModel(config).to(config.DEVICE)
            m.load_state_dict(torch.load(cp, map_location=config.DEVICE, weights_only=True))
            m.eval()
            ensemble_models.append(m)
        else:
            logger.info(f"[WARN] Không tìm thấy checkpoint: {cp}")

    logger.info(f"Đã load {len(ensemble_models)}/{config.K_FOLDS} fold checkpoints cho ensemble")

    if len(ensemble_models) == 0:
        logger.info("❌ Không có checkpoint nào để ensemble test.")
    else:
        all_labels_test, all_preds_test = [], []
        with torch.no_grad():
            for batch_data in test_loader:
                if batch_data is None:
                    continue
                imgs, labels, _ = batch_data
                imgs = imgs.to(config.DEVICE)
                fold_preds = [m(imgs).squeeze(-1).cpu().numpy() for m in ensemble_models]
                ensemble_pred = np.mean(fold_preds, axis=0)
                all_labels_test.extend(labels.numpy())
                all_preds_test.extend(ensemble_pred.tolist())

        test_r2 = r2_score(all_labels_test, all_preds_test)
        test_rmse = np.sqrt(mean_squared_error(all_labels_test, all_preds_test))
        test_mae = mean_absolute_error(all_labels_test, all_preds_test)

        logger.info(f"ENSEMBLE TEST | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R2: {test_r2:.4f}")

        # --- Ghi ensemble test metrics -> CSV ---
        with open(test_log_path, "a", encoding="utf-8") as f:
            f.write(f"{test_r2:.8f},{test_rmse:.8f},{test_mae:.8f}\n")

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(all_labels_test, all_preds_test, alpha=0.5, s=25, color='darkorange', label='Ensemble Samples')
        _mn = min(min(all_labels_test), min(all_preds_test))
        _mx = max(max(all_labels_test), max(all_preds_test))
        ax.plot([_mn, _mx], [_mn, _mx], 'r--', linewidth=1.5, label='Perfect fit')
        ax.set_xlabel("Actual Pain Score")
        ax.set_ylabel("Predicted Pain Score")
        ax.set_title(f"ENSEMBLE TEST | R²={test_r2:.3f}  RMSE={test_rmse:.3f}  MAE={test_mae:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        scatter_path = os.path.join(export_base, "scatter_ensemble_test.png")
        fig.savefig(scatter_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"✅ Đã lưu biểu đồ ensemble test tại: {scatter_path}")

    results_df = pd.DataFrame(fold_results)
    if not results_df.empty:
        mean_r2 = results_df['val_r2'].mean()
        std_r2 = results_df['val_r2'].std()
        mean_rmse = results_df['val_rmse'].mean()
        std_rmse = results_df['val_rmse'].std()
        mean_mae = results_df['val_mae'].mean()
        std_mae = results_df['val_mae'].std()

        best_fold = results_df.loc[results_df['val_r2'].idxmax()]

        logger.info("\n" + "="*80)
        logger.info(f"--- V11 FINAL RESULTS ({config.K_FOLDS}-Fold Stratified) ---")
        logger.info("="*80)
        logger.info(f"{'fold':>6} {'r2':>15} {'rmse':>15} {'mae':>15}")

        for _, row in results_df.iterrows():
            logger.info(f"{int(row['fold']):>6} {row['val_r2']:>15.6f} {row['val_rmse']:>15.6f} {row['val_mae']:>15.6f}")

        logger.info("")
        logger.info(f"Mean RMSE: {mean_rmse:.4f} (±{std_rmse:.4f})")
        logger.info(f"Mean MAE : {mean_mae:.4f} (±{std_mae:.4f})")
        logger.info(f"Mean R²  : {mean_r2:.4f} (±{std_r2:.4f})")
        logger.info(f"🏆 Best Fold: Fold {int(best_fold['fold'])} (R²: {best_fold['val_r2']:.4f}, RMSE: {best_fold['val_rmse']:.4f})")
        logger.info("")

        logger.info("v11 Improvements Target:")
        r2_check = "✓" if mean_r2 >= 0.68 else "x"
        rmse_check = "✓" if mean_rmse <= 0.62 else "x"
        std_check = "✓" if std_r2 <= 0.04 else "x"

        logger.info(f"{r2_check}   V10 Mean R²: 0.6106   -> V11 Target: ≥0.68-0.70  | Đạt được: {mean_r2:.4f}")
        logger.info(f"{rmse_check} V10 Mean RMSE: 0.6823 -> V11 Target: ≤0.60-0.62 | Đạt được: {mean_rmse:.4f}")
        logger.info(f"{std_check}  V10 Std Dev: ±7%      -> V11 Target: ±3-4%       | Đạt được: ±{std_r2*100:.2f}%")
        logger.info("="*80)
        logger.info(f"✅ HOÀN TẤT! File log được lưu tại: {log_file_path}")
        logger.info(f"✅ Epoch CSV: {epoch_log_path}")
        logger.info(f"✅ Fold  CSV: {fold_log_path}")
        logger.info(f"✅ Test  CSV: {test_log_path}")


if __name__ == '__main__':
    if sys.platform == "win32":
        cfg.NUM_WORKERS = 0
    main(cfg)