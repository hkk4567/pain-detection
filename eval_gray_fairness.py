# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import timm
import torchvision.transforms as transforms
from collections import Counter
import cv2
import random

# ================================================================
# [QUAN TRỌNG] TÊN FOLDER CHỨA CHECKPOINT GRAY ĐÃ TRAIN
# ================================================================
TRAINED_RUN_TIMESTAMP = "2026-04-15_11-21-45_GRAY_PureRaw_Baseline"

# ================================================================
# CẤU HÌNH CƠ BẢN
# ================================================================
class Config:
    DATA_ROOT_DIR   = "dataset_osfstorage-archive/Stimuli"
    LABELS_PATH     = "dataset_osfstorage-archive/NormingData/DFD_NormingData.csv"
    SKIN_COLOR_BASE = os.path.join("dataset_osfstorage-archive", "Color Skin")
    
    BATCH_SIZE  = 16
    IMG_SIZE    = 300
    K_FOLDS       = 10
    CONFIDENCE_THRESHOLD    = 0.8   
    BELIEVABILITY_THRESHOLD = 2.5   
    BACKBONE      = 'efficientnet_b3'
    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED          = 42
    NUM_WORKERS   = 4 if sys.platform != "win32" else 0

cfg = Config()

# ================================================================
# HÀM BỔ TRỢ
# ================================================================
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    imgs   = torch.stack([x[0] for x in batch])
    labels = torch.stack([x[1] for x in batch])
    groups = [x[2] for x in batch]
    return imgs, labels, groups

def analyze_fairness_by_race(all_labels, all_preds, all_races, logger):
    logger.info("\n" + "="*70)
    logger.info(f"📊 PHÂN TÍCH FAIRNESS TRÊN TẬP TEST (RUN: {TRAINED_RUN_TIMESTAMP})")
    logger.info("="*70)
    
    results_by_race = {'Asian': {'y_true': [], 'y_pred': []},
                       'Black': {'y_true': [], 'y_pred': []},
                       'White': {'y_true': [], 'y_pred': []}}
                       
    for y_t, y_p, race in zip(all_labels, all_preds, all_races):
        if race in results_by_race:
            results_by_race[race]['y_true'].append(y_t)
            results_by_race[race]['y_pred'].append(y_p)
            
    logger.info(f"{'Nhóm da':<10} | {'Số mẫu':<8} | {'MAE':<10} | {'RMSE':<10} | {'Bias (Pred - True)':<20}")
    logger.info("-" * 70)
    
    for race in ['Asian', 'Black', 'White']:
        y_t = np.array(results_by_race[race]['y_true'])
        y_p = np.array(results_by_race[race]['y_pred'])
        
        if len(y_t) > 0:
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
            bias = np.mean(y_p - y_t)
            logger.info(f"{race:<10} | {len(y_t):<8} | {mae:<10.4f} | {rmse:<10.4f} | {bias:<20.4f}")
    logger.info("="*70)


# ================================================================
# SMART FACE CLAHE & DATASET
# ================================================================
class SmartFaceCLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    def __call__(self, img):
        if not isinstance(img, Image.Image): return img
        img_np = np.array(img.convert("RGB"))
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        lab_eq = cv2.merge((self.clahe.apply(l), a, b))   
        return Image.fromarray(cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB))

class PainDataset(Dataset):
    def __init__(self, config):
        self.cfg = config
        self.image_path_map = self._create_smart_file_map(self.cfg.DATA_ROOT_DIR)
        self.skin_color_map = self._create_skin_color_map(self.cfg.SKIN_COLOR_BASE)
        df = pd.read_csv(self.cfg.LABELS_PATH)
        df.columns = df.columns.str.strip()
        num_cols = ['Pain_Expression', 'PhysicalPain_Neutral', 'HowBelievable', 'OpenFace_confidence', 'Male']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        if 'OpenFace_confidence' in df.columns:
            df = df[df['OpenFace_confidence'] >= self.cfg.CONFIDENCE_THRESHOLD]
        if 'HowBelievable' in df.columns:
            df = df[df['HowBelievable'].fillna(9.0) >= self.cfg.BELIEVABILITY_THRESHOLD]

        self.data_list, self.stratify_labels, self.groups = [], [], []
        for _, row in df.iterrows():
            target = str(row['Target']).strip()
            clean_name = os.path.splitext(target)[0].lower().replace('_cropped', '').replace('_standardized', '').replace('_unedited', '')
            parts = target.split('_')
            subject_id = parts[2].upper() if len(parts) >= 3 else clean_name
            actual_key = clean_name if clean_name in self.image_path_map else (clean_name + "_earring" if (clean_name + "_earring") in self.image_path_map else None)
            if actual_key is None or actual_key not in self.skin_color_map: continue
            
            p_val, n_val = row['Pain_Expression'], row['PhysicalPain_Neutral']
            if not pd.isna(p_val): label = float(p_val)
            elif not pd.isna(n_val): label = float(n_val)
            else: continue
            
            race = self.skin_color_map[actual_key]   
            group_id = f"{race}_{'Male' if row.get('Male') == 1 else 'Female'}"
            self.data_list.append({'path': self.image_path_map[actual_key], 'label': label, 'group': group_id, 'race': race})
            self.stratify_labels.append(group_id)
            self.groups.append(subject_id) 

    def _create_smart_file_map(self, root_dir):
        file_map = {}
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    clean_name = os.path.splitext(f)[0].lower().replace('_cropped', '').replace('_standardized', '').replace('_unedited', '')
                    if "cropped" in dirpath.lower() or clean_name not in file_map:
                        file_map[clean_name] = os.path.join(dirpath, f)
        return file_map

    def _create_skin_color_map(self, skin_base):
        color_map = {}
        if not os.path.exists(skin_base): return color_map
        mapping = {'black': 'Black', 'white': 'White', 'yellow': 'Asian'}
        for folder, label in mapping.items():
            folder_path = os.path.join(skin_base, folder)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        clean_name = os.path.splitext(f)[0].lower().replace('_cropped', '').replace('_standardized', '').replace('_unedited', '')
                        color_map[clean_name] = label
        return color_map

    def __len__(self): return len(self.data_list)


class SplitTransformDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base, self.transform = base_dataset, transform
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        item = self.base.data_list[idx]
        try:
            img = ImageOps.exif_transpose(Image.open(item['path'])).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, torch.tensor(item['label'], dtype=torch.float), item['group']
        except Exception: return None


# ================================================================
# MODEL 
# ================================================================
class BaselinePainModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Pretrained = False vì chúng ta chỉ load trọng số từ file Checkpoint
        self.backbone = timm.create_model(cfg.BACKBONE, pretrained=False, num_classes=0, global_pool='avg')
        self.pain_head = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(self.backbone.num_features, 1))
    def forward(self, x): return self.pain_head(self.backbone(x))


# ================================================================
# CHƯƠNG TRÌNH CHÍNH (ĐÁNH GIÁ ĐỘC LẬP MODEL GRAY)
# ================================================================
def main():
    seed_everything(cfg.SEED)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('eval_gray')

    logger.info("⏳ Đang chuẩn bị dữ liệu (Eval Mode - GRAY Model)...")
    
    # [QUAN TRỌNG] CẤU HÌNH TRANSFORM DÀNH RIÊNG CHO ẢNH XÁM
    eval_transform = transforms.Compose([
        SmartFaceCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3), # Chuyển sang ảnh Xám 3 kênh
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.449, 0.449, 0.449], std=[0.226, 0.226, 0.226]) # Chuẩn hóa Gray
    ])

    full_dataset = PainDataset(cfg)
    eval_dataset = SplitTransformDataset(full_dataset, eval_transform)

    if len(full_dataset) == 0:
        logger.info("❌ Không tìm thấy dữ liệu ảnh gốc!")
        return

    # =========================================================================
    # TÁI TẠO LẠI CHÍNH XÁC TẬP TEST LÚC TRAIN (CỐ ĐỊNH THEO SUBJECT ID)
    # =========================================================================
    all_indices = np.arange(len(full_dataset))
    all_labels  = np.array(full_dataset.stratify_labels)
    
    counts = Counter(all_labels)
    rare_groups = [g for g, c in counts.items() if c < 2]
    if rare_groups:
        mask = np.isin(all_labels, rare_groups, invert=True)
        all_indices = all_indices[mask]

    subject_to_race = {}
    for idx in all_indices:
        subj = full_dataset.groups[idx]
        race = full_dataset.data_list[idx]['race']
        subject_to_race[subj] = race

    asian_subjects = [s for s, r in subject_to_race.items() if r == 'Asian']
    black_subjects = [s for s, r in subject_to_race.items() if r == 'Black']
    white_subjects = [s for s, r in subject_to_race.items() if r == 'White']

    rng = random.Random(cfg.SEED)
    rng.shuffle(asian_subjects)
    rng.shuffle(black_subjects)
    rng.shuffle(white_subjects)

    # 6 Asian, 12 Black, 18 White 
    test_subjects = set(asian_subjects[:6] + black_subjects[:12] + white_subjects[:18])
    test_idx = np.array([idx for idx in all_indices if full_dataset.groups[idx] in test_subjects])
    
    test_loader = DataLoader(
        Subset(eval_dataset, test_idx), batch_size=cfg.BATCH_SIZE, shuffle=False, 
        collate_fn=custom_collate, num_workers=cfg.NUM_WORKERS, pin_memory=True
    )

    # =========================================================================
    # LOAD ENSEMBLE MODELS (GRAY) TỪ FOLDER CHECKPOINTS
    # =========================================================================
    ensemble_models = []
    for i in range(cfg.K_FOLDS):
        cp_path = os.path.join("checkpoints", f"{TRAINED_RUN_TIMESTAMP}_fold_{i+1}.pth")
        if os.path.exists(cp_path):
            m = BaselinePainModel(cfg).to(cfg.DEVICE)
            m.load_state_dict(torch.load(cp_path, map_location=cfg.DEVICE, weights_only=True))
            m.eval()
            ensemble_models.append(m)
        else:
            logger.warning(f"⚠️ Không tìm thấy checkpoint Fold {i+1}: {cp_path}")

    if not ensemble_models:
        logger.error(f"❌ Không tìm thấy bất kỳ checkpoint nào của run '{TRAINED_RUN_TIMESTAMP}'. Hãy đảm bảo folder 'checkpoints' nằm đúng vị trí!")
        return

    logger.info(f"✅ Đã tải thành công {len(ensemble_models)}/{cfg.K_FOLDS} models. Bắt đầu đánh giá Ensemble...")

    # =========================================================================
    # CHẠY INFERENCE VÀ TÍNH TOÁN FAIRNESS
    # =========================================================================
    all_labels_te, all_preds_te = [], []
    with torch.no_grad():
        for batch_data in test_loader:
            if batch_data is None: continue
            imgs, labels, _ = batch_data
            imgs = imgs.to(cfg.DEVICE)
            ens_pred = np.mean([m(imgs).squeeze(-1).cpu().numpy() for m in ensemble_models], axis=0)
            all_labels_te.extend(labels.numpy())
            all_preds_te.extend(ens_pred.tolist())

    te_r2   = r2_score(all_labels_te, all_preds_te)
    te_rmse = np.sqrt(mean_squared_error(all_labels_te, all_preds_te))
    te_mae  = mean_absolute_error(all_labels_te, all_preds_te)
    
    logger.info("\n" + "="*70)
    logger.info(f"TỔNG QUAN ENSEMBLE GRAY (Tất cả nhóm) | RMSE: {te_rmse:.4f} | MAE: {te_mae:.4f} | R²: {te_r2:.4f}")
    
    # GỌI HÀM PHÂN TÍCH FAIRNESS
    test_races = [full_dataset.data_list[i]['race'] for i in test_idx]
    analyze_fairness_by_race(all_labels_te, all_preds_te, test_races, logger)

if __name__ == '__main__':
    main()