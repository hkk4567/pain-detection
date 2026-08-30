# -- coding: utf-8 --
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
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
import re

# ================================================================
# SMART FACE CLAHE 
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
        lab_eq = cv2.merge((self.clahe.apply(l), a, b))   
        return Image.fromarray(cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB))


# ================================================================
# PHÂN LOẠI ĐAU & HẰNG SỐ
# ================================================================
def categorize_pain(score):
    if score < 2.0:  return 0
    elif score < 4.0: return 1
    return 2

PAIN_LABELS = ["Low", "Medium", "High"]


# ================================================================
# CẤU HÌNH V11 — ABLATION (RGB UNWEIGHTED BASELINE)
# ================================================================
class Config:
    DATA_ROOT_DIR   = "dataset_osfstorage-archive/Stimuli"
    LABELS_PATH     = "dataset_osfstorage-archive/NormingData/DFD_NormingData.csv"
    SKIN_COLOR_BASE = os.path.join("dataset_osfstorage-archive", "Color Skin")

    BATCH_SIZE  = 16
    IMG_SIZE    = 300
    LEARNING_RATE = 5.0e-4
    EPOCHS_KFOLD  = 50
    K_FOLDS       = 10
    TEST_SPLIT    = 0.15

    EARLY_STOPPING_PATIENCE = 15
    CONFIDENCE_THRESHOLD    = 0.8   
    BELIEVABILITY_THRESHOLD = 2.5   
    BACKBONE      = 'efficientnet_b3'
    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED          = 42
    WARMUP_EPOCHS = 5
    
    # [ABLATION] Đổi tên thư mục log để nhận diện đây là bản không trọng số
    RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_V11_RGB_UNWEIGHTED_Baseline"
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

# [ABLATION] Thêm hàm tự động phân tích Fairness để đưa vào báo cáo
def analyze_fairness_by_race(all_labels, all_preds, all_races, logger):
    logger.info("\n" + "="*60)
    logger.info("PHÂN TÍCH FAIRNESS (SAI SỐ THEO MÀU DA - UNWEIGHTED)")
    logger.info("="*60)
    
    results_by_race = {'Asian': {'y_true': [], 'y_pred': []},
                       'Black': {'y_true': [], 'y_pred': []},
                       'White': {'y_true': [], 'y_pred': []}}
                       
    for y_t, y_p, race in zip(all_labels, all_preds, all_races):
        if race in results_by_race:
            results_by_race[race]['y_true'].append(y_t)
            results_by_race[race]['y_pred'].append(y_p)
            
    logger.info(f"{'Nhóm da':<10} | {'Số mẫu':<8} | {'MAE':<10} | {'RMSE':<10} | {'Bias (Pred - True)':<20}")
    logger.info("-" * 60)
    
    for race in ['Asian', 'Black', 'White']:
        y_t = np.array(results_by_race[race]['y_true'])
        y_p = np.array(results_by_race[race]['y_pred'])
        
        if len(y_t) > 0:
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
            bias = np.mean(y_p - y_t)
            
            logger.info(f"{race:<10} | {len(y_t):<8} | {mae:<10.4f} | {rmse:<10.4f} | {bias:<20.4f}")
    logger.info("="*60)


# ================================================================
# DATASET
# ================================================================
class PainDataset(Dataset):
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
        self.groups          = [] 

        n_from_pain_expr    = 0
        n_from_phys_neutral = 0
        n_dropped_nan_label = 0

        for _, row in df.iterrows():
            target     = str(row['Target']).strip()
            base_name  = os.path.splitext(target)[0].lower()
            clean_name = base_name.replace('_cropped', '').replace('_standardized', '').replace('_unedited', '')

            parts = target.split('_')
            if len(parts) >= 3:
                subject_id = parts[2].upper() 
            else:
                subject_id = clean_name 

            actual_key = None
            if clean_name in self.image_path_map:
                actual_key = clean_name
            elif (clean_name + "_earring") in self.image_path_map:
                actual_key = clean_name + "_earring"

            if actual_key is None:                       continue
            if actual_key not in self.skin_color_map:    continue

            full_path = self.image_path_map[actual_key]

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
                continue  

            race = self.skin_color_map[actual_key]   
            gender   = "Male" if row.get('Male') == 1 else "Female"
            group_id = f"{race}_{gender}"

            self.data_list.append({'path': full_path, 'label': label, 'group': group_id, 'race': race})
            self.stratify_labels.append(group_id)
            self.groups.append(subject_id) 

        logging.info(f"[Label source] Pain_Expression: {n_from_pain_expr} | "
                     f"PhysicalPain_Neutral (fallback): {n_from_phys_neutral} | "
                     f"Dropped: {n_dropped_nan_label}")

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
        item = self.data_list[idx]
        try:
            img = ImageOps.exif_transpose(Image.open(item['path'])).convert("RGB")
            return img, torch.tensor(item['label'], dtype=torch.float), item['group']
        except Exception as e:
            logging.warning(f"[DataError] {item['path']} | {e}")
            return None


class SplitTransformDataset(Dataset):
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

    epoch_log_path = os.path.join(export_base, "epoch_metrics.csv")
    fold_log_path  = os.path.join(export_base, "fold_metrics.csv")
    test_log_path  = os.path.join(export_base, "ensemble_test_metrics.csv")

    with open(epoch_log_path, "w", encoding="utf-8") as f:
        f.write("fold,epoch,train_loss,val_loss,lr\n")
    with open(fold_log_path, "w", encoding="utf-8") as f:
        f.write("fold,val_r2,val_rmse,val_mae\n")
    with open(test_log_path, "w", encoding="utf-8") as f:
        f.write("ensemble_test_r2,ensemble_test_rmse,ensemble_test_mae\n")

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

    full_dataset  = PainDataset(config)
    train_dataset = SplitTransformDataset(full_dataset, train_transform)
    eval_dataset  = SplitTransformDataset(full_dataset, eval_transform)

    if len(full_dataset) == 0:
        logger.info("❌ Không có dữ liệu để huấn luyện!")
        return

    logger.info("\n" + "="*60)
    logger.info("🚀 BẮT ĐẦU HUẤN LUYỆN MODEL V11 (RGB UNWEIGHTED BASELINE)")
    logger.info("⚠️ CHẾ ĐỘ ABLATION: Đã TẮT Weighted Loss. Sử dụng MSE tiêu chuẩn.")
    logger.info("="*60)
    logger.info(f"🖥️  Thiết bị   : {str(config.DEVICE).upper()}")
    logger.info(f"📂  Run        : {config.RUN_TIMESTAMP}")
    logger.info(f"📊  Cấu hình   : {config.K_FOLDS}-Folds | Test: {config.TEST_SPLIT*100:.0f}%")
    logger.info(f"📁  Skin map   : {len(full_dataset.skin_color_map)} ảnh (Black/White/Asian)")
    logger.info(f"📈  Tổng mẫu   : {len(full_dataset)} (sau lọc confidence & believability)")
    logger.info(f"👤  Số Subjects: {len(set(full_dataset.groups))}") 
    logger.info("="*60)

    all_indices = np.arange(len(full_dataset))
    all_labels  = np.array(full_dataset.stratify_labels)
    all_groups  = np.array(full_dataset.groups) 

    counts      = Counter(all_labels)
    rare_groups = [g for g, c in counts.items() if c < 2]
    if rare_groups:
        mask        = np.isin(all_labels, rare_groups, invert=True)
        all_indices = all_indices[mask]
        all_labels  = all_labels[mask]
        all_groups  = all_groups[mask] 

    subject_to_race = {}
    for idx in all_indices:
        subj = full_dataset.groups[idx]
        race = full_dataset.data_list[idx]['race']
        subject_to_race[subj] = race

    asian_subjects = [s for s, r in subject_to_race.items() if r == 'Asian']
    black_subjects = [s for s, r in subject_to_race.items() if r == 'Black']
    white_subjects = [s for s, r in subject_to_race.items() if r == 'White']

    import random
    rng = random.Random(config.SEED)
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

    train_val_idx_list = []
    test_idx_list = []
    for idx in all_indices:
        if full_dataset.groups[idx] in test_subjects:
            test_idx_list.append(idx)
        else:
            train_val_idx_list.append(idx)

    train_val_idx = np.array(train_val_idx_list)
    test_idx = np.array(test_idx_list)

    test_loader = DataLoader(
        Subset(eval_dataset, test_idx),
        batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

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
    for r in all_races:
        tr_c, tr_p = tv_dist.get(r, (0, 0.0))
        te_c, te_p = te_dist.get(r, (0, 0.0))
        line = f"{r:<15} | {tr_c:>12} | {tr_p:>7.2f}% | {te_c:>6} | {te_p:>7.2f}%"
        logger.info(line)
    logger.info("-" * 60)

    sgkf = StratifiedGroupKFold(n_splits=config.K_FOLDS, shuffle=True, random_state=config.SEED)
    y_train_val = [full_dataset.stratify_labels[i] for i in train_val_idx]
    groups_train_val = [full_dataset.groups[i] for i in train_val_idx] 
    
    fold_results = []
    
    # Khởi tạo hàm loss tiêu chuẩn
    mse_criterion = nn.MSELoss()

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
                imgs, labels, _ = batch_data
                imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)

                preds = model(imgs).squeeze(-1)
                
                # [ABLATION] Dùng hàm Loss tiêu chuẩn thay cho Weighted Loss
                loss = mse_criterion(preds, labels)

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
                    imgs, labels, _ = batch_data
                    imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
                    
                    preds = model(imgs).squeeze(-1)
                    
                    # [ABLATION] Dùng hàm Loss tiêu chuẩn
                    batch_loss = mse_criterion(preds, labels)
                    
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

        # [ABLATION] GỌI HÀM PHÂN TÍCH FAIRNESS TỰ ĐỘNG
        test_races = [full_dataset.data_list[i]['race'] for i in test_idx]
        analyze_fairness_by_race(all_labels_te, all_preds_te, test_races, logger)

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

        logger.info("\n" + "="*60)
        logger.info(f"--- V11 RGB UNWEIGHTED FINAL RESULTS ({config.K_FOLDS}-Fold Stratified) ---")
        logger.info("="*60)
        
        logger.info(f"\nMean RMSE : {mean_rmse:.4f} (±{std_rmse:.4f})")
        logger.info(f"Mean MAE  : {mean_mae:.4f}  (±{std_mae:.4f})")
        logger.info(f"Mean R²   : {mean_r2:.4f}  (±{std_r2:.4f})")
        logger.info("="*60)
        logger.info(f"✅ HOÀN TẤT! Log: {log_file_path}")

if __name__ == '__main__':
    if sys.platform == "win32":
        cfg.NUM_WORKERS = 0
    main(cfg)