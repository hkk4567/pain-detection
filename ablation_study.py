# -- coding: utf-8 --
import os
import sys
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime
import timm
import torchvision.transforms as transforms
from collections import Counter
import cv2
from tqdm import tqdm  # THƯ VIỆN THEO DÕI TIẾN ĐỘ

# ================================================================
# SMART FACE CLAHE
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

# ================================================================
# CẤU HÌNH GỐC
# ================================================================
class BaseConfig:
    DATA_ROOT_DIR   = "dataset_osfstorage-archive/Stimuli"
    LABELS_PATH     = "dataset_osfstorage-archive/NormingData/DFD_NormingData.csv"
    SKIN_COLOR_BASE = os.path.join("dataset_osfstorage-archive", "Color Skin")

    BATCH_SIZE  = 16
    IMG_SIZE    = 300
    LEARNING_RATE = 5.0e-4
    
    # -------------------------------------------------------------
    # THÔNG SỐ TRAIN (Có thể chỉnh lên 10-folds và 50 epochs nếu máy đủ mạnh)
    # -------------------------------------------------------------
    EPOCHS_KFOLD  = 30 
    K_FOLDS       = 5  
    EARLY_STOPPING_PATIENCE = 10
    
    CONFIDENCE_THRESHOLD    = 0.8
    BELIEVABILITY_THRESHOLD = 2.5
    BACKBONE      = 'efficientnet_b3'
    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED          = 42
    WARMUP_EPOCHS = 3
    RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_Ablation"
    NUM_WORKERS   = 4 if sys.platform != "win32" else 0

# ================================================================
# CÁC KỊCH BẢN THÍ NGHIỆM (ABLATION EXPERIMENTS)
# ================================================================
EXPERIMENTS = [
    {
        "exp_id": "01_Baseline_Gray",
        "USE_RGB": False, "USE_CLAHE": False, "USE_WEIGHTED_LOSS": False
    },
    {
        "exp_id": "02_RGB_NoCLAHE",
        "USE_RGB": True,  "USE_CLAHE": False, "USE_WEIGHTED_LOSS": False
    },
    {
        "exp_id": "03_RGB_CLAHE",
        "USE_RGB": True,  "USE_CLAHE": True,  "USE_WEIGHTED_LOSS": False
    },
    {
        "exp_id": "04_Full_Fairness",
        "USE_RGB": True,  "USE_CLAHE": True,  "USE_WEIGHTED_LOSS": True
    }
]

# ================================================================
# HÀM BỔ TRỢ & DATASET
# ================================================================
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_weights(dataset_list):
    groups = [item['group'] for item in dataset_list]
    count = Counter(groups)
    total = len(groups)
    n_cls = len(count)
    return {group: (total / (n_cls * freq)) for group, freq in count.items()}

def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    imgs = torch.stack([x[0] for x in batch])
    labels = torch.stack([x[1] for x in batch])
    groups = [x[2] for x in batch]
    return imgs, labels, groups

class PainDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.image_path_map = self._create_file_map(cfg.DATA_ROOT_DIR)
        self.skin_color_map = self._create_skin_map(cfg.SKIN_COLOR_BASE)
        df = pd.read_csv(cfg.LABELS_PATH)
        df.columns = df.columns.str.strip()
        for col in ['Pain_Expression', 'PhysicalPain_Neutral', 'HowBelievable', 'OpenFace_confidence', 'Male']:
            if col in df.columns: df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        if 'OpenFace_confidence' in df.columns: df = df[df['OpenFace_confidence'] >= cfg.CONFIDENCE_THRESHOLD]
        if 'HowBelievable' in df.columns: df = df[df['HowBelievable'].fillna(9.0) >= cfg.BELIEVABILITY_THRESHOLD]

        self.data_list, self.stratify_labels, self.groups = [], [], []
        for _, row in df.iterrows():
            target = str(row['Target']).strip()
            clean_name = os.path.splitext(target)[0].lower().replace('_cropped', '').replace('_standardized', '').replace('_unedited', '')
            subj_id = target.split('_')[2].upper() if len(target.split('_')) >= 3 else clean_name
            actual_key = clean_name if clean_name in self.image_path_map else clean_name + "_earring" if (clean_name + "_earring") in self.image_path_map else None
            
            if not actual_key or actual_key not in self.skin_color_map: continue
            
            p_val, n_val = row['Pain_Expression'], row['PhysicalPain_Neutral']
            if not pd.isna(p_val): label = float(p_val)
            elif not pd.isna(n_val): label = float(n_val)
            else: continue

            race = self.skin_color_map[actual_key]
            gender = "Male" if row.get('Male') == 1 else "Female"
            group_id = f"{race}_{gender}"

            self.data_list.append({'path': self.image_path_map[actual_key], 'label': label, 'group': group_id, 'race': race})
            self.stratify_labels.append(group_id)
            self.groups.append(subj_id)

    def _create_file_map(self, root_dir):
        fm = {}
        for dp, _, fns in os.walk(root_dir):
            for f in fns:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    cn = os.path.splitext(f)[0].lower().replace('_cropped', '').replace('_standardized', '').replace('_unedited', '')
                    if "cropped" in dp.lower() or cn not in fm: fm[cn] = os.path.join(dp, f)
        return fm

    def _create_skin_map(self, skin_base):
        cm = {}
        if not os.path.exists(skin_base): return cm
        for folder, label in {'black': 'Black', 'white': 'White', 'yellow': 'Asian'}.items():
            folder_path = os.path.join(skin_base, folder)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        cn = os.path.splitext(f)[0].lower().replace('_cropped', '').replace('_standardized', '').replace('_unedited', '')
                        cm[cn] = label
        return cm

    def __len__(self): return len(self.data_list)

class SplitTransformDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        item = self.base.data_list[idx]
        try:
            img = ImageOps.exif_transpose(Image.open(item['path'])).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, torch.tensor(item['label'], dtype=torch.float), item['group']
        except Exception: return None

# ================================================================
# MODEL & SCHEDULER
# ================================================================
class BaselinePainModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(cfg.BACKBONE, pretrained=True, num_classes=0, global_pool='avg')
        self.pain_head = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(self.backbone.num_features, 1))
    def forward(self, x):
        return self.pain_head(self.backbone(x))

class WarmupCosineAnnealingLR:
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.optimizer, self.warmup_epochs, self.total_epochs = optimizer, warmup_epochs, total_epochs
        self.base_lr, self.epoch = optimizer.defaults['lr'], 0
    def step(self):
        lr = self.base_lr * (self.epoch + 1) / self.warmup_epochs if self.epoch < self.warmup_epochs else \
             self.base_lr * (1 + np.cos(np.pi * (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs))) / 2
        for pg in self.optimizer.param_groups: pg['lr'] = lr
        self.epoch += 1

# ================================================================
# HÀM CHẠY 1 THÍ NGHIỆM ĐƠN LẺ
# ================================================================
def run_single_experiment(exp_config, cfg, full_dataset, train_val_idx, test_idx, kfold_splits, logger, export_base):
    logger.info(f"\n" + "🚀"*20)
    logger.info(f"ĐANG CHẠY THÍ NGHIỆM: {exp_config['exp_id']}")
    logger.info(f"Cấu hình: RGB={exp_config['USE_RGB']} | CLAHE={exp_config['USE_CLAHE']} | WeightLoss={exp_config['USE_WEIGHTED_LOSS']}")
    logger.info("🚀"*20)

    # 1. Tạo Transforms dựa trên Cấu hình
    RGB_MEAN, RGB_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    train_tr_list, eval_tr_list = [], []
    if exp_config['USE_CLAHE']:
        clahe = SmartFaceCLAHE()
        train_tr_list.append(clahe)
        eval_tr_list.append(clahe)

    train_tr_list.append(transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)))
    eval_tr_list.append(transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)))

    if not exp_config['USE_RGB']:
        train_tr_list.append(transforms.Grayscale(num_output_channels=3))
        eval_tr_list.append(transforms.Grayscale(num_output_channels=3))

    train_tr_list.extend([
        transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(), transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
    ])
    eval_tr_list.extend([transforms.ToTensor(), transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)])

    train_dataset = SplitTransformDataset(full_dataset, transforms.Compose(train_tr_list))
    eval_dataset  = SplitTransformDataset(full_dataset, transforms.Compose(eval_tr_list))

    test_loader = DataLoader(Subset(eval_dataset, test_idx), batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=custom_collate, num_workers=cfg.NUM_WORKERS)

    fold_results = []
    ensemble_models = []

    # 2. Huấn luyện qua các Fold
    for fold, (tr_rel, val_rel) in enumerate(kfold_splits):
        train_ids, val_ids = train_val_idx[tr_rel], train_val_idx[val_rel]
        group_weights = calculate_weights([full_dataset.data_list[i] for i in train_ids])
        
        train_loader = DataLoader(Subset(train_dataset, train_ids), batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=custom_collate, num_workers=cfg.NUM_WORKERS)
        val_loader   = DataLoader(Subset(eval_dataset, val_ids), batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=custom_collate, num_workers=cfg.NUM_WORKERS)

        model = BaselinePainModel(cfg).to(cfg.DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
        scheduler = WarmupCosineAnnealingLR(optimizer, cfg.WARMUP_EPOCHS, cfg.EPOCHS_KFOLD)
        scheduler.step()

        best_val_loss, patience_counter = float('inf'), 0
        cp_path = os.path.join(export_base, f"{exp_config['exp_id']}_fold_{fold+1}.pth")

        logger.info(f"\n--- FOLD {fold+1}/{cfg.K_FOLDS} ---")
        
        for epoch in range(cfg.EPOCHS_KFOLD):
            model.train()
            train_loss_sum, train_n = 0.0, 0
            
            # THANH TIẾN TRÌNH CỦA BATCH (SẼ HIỂN THỊ KHI CHẠY)
            pbar = tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{cfg.EPOCHS_KFOLD}", leave=False)
            
            for batch in pbar:
                if not batch: continue
                imgs, labels, b_groups = batch
                imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                preds = model(imgs).squeeze(-1)
                
                if exp_config['USE_WEIGHTED_LOSS']:
                    b_w = torch.tensor([group_weights[g] for g in b_groups], dtype=preds.dtype, device=cfg.DEVICE)
                    loss = ((preds - labels) ** 2 * b_w).mean()
                else:
                    loss = nn.MSELoss()(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * labels.size(0)
                train_n += labels.size(0)
                pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

            model.eval()
            val_loss_sum, val_n = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    if not batch: continue
                    imgs, labels, _ = batch
                    imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                    preds = model(imgs).squeeze(-1)
                    val_loss_sum += nn.MSELoss(reduction='sum')(preds, labels).item()
                    val_n += labels.size(0)
            
            avg_train = train_loss_sum / train_n if train_n > 0 else 0.0
            avg_val = val_loss_sum / val_n if val_n > 0 else 0.0
            cur_lr = optimizer.param_groups[0]['lr']

            # IN THÔNG SỐ CỦA EPOCH
            epoch_log_str = f"   Ep {epoch+1:02d}/{cfg.EPOCHS_KFOLD} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {cur_lr:.6f}"

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                torch.save(model.state_dict(), cp_path)
                logger.info(epoch_log_str + f" (Best Val: {best_val_loss:.4f}) [Saved]")
            else:
                patience_counter += 1
                logger.info(epoch_log_str + f" (No improve: {patience_counter}/{cfg.EARLY_STOPPING_PATIENCE})")
                if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                    logger.info("   ⏹️ Early stopping!")
                    break
            scheduler.step()

        # Đánh giá Fold hiện tại
        model.load_state_dict(torch.load(cp_path, weights_only=True))
        model.eval()
        all_l, all_p = [], []
        with torch.no_grad():
            for batch in val_loader:
                if not batch: continue
                imgs, labels, _ = batch
                preds = model(imgs.to(cfg.DEVICE)).squeeze(-1)
                all_l.extend(labels.numpy()); all_p.extend(preds.cpu().numpy())
        
        r2, rmse, mae = r2_score(all_l, all_p), np.sqrt(mean_squared_error(all_l, all_p)), mean_absolute_error(all_l, all_p)
        fold_results.append({'fold': fold+1, 'r2': r2, 'rmse': rmse, 'mae': mae})
        logger.info(f"   => Fold {fold+1} Result: R²={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f}")
        ensemble_models.append(model)

    # 3. Đánh giá Ensemble trên Test Set
    all_l_te, all_p_te = [], []
    with torch.no_grad():
        for batch in test_loader:
            if not batch: continue
            imgs, labels, _ = batch
            imgs = imgs.to(cfg.DEVICE)
            ens_pred = np.mean([m(imgs).squeeze(-1).cpu().numpy() for m in ensemble_models], axis=0)
            all_l_te.extend(labels.numpy()); all_p_te.extend(ens_pred)
    
    ens_r2, ens_rmse, ens_mae = r2_score(all_l_te, all_p_te), np.sqrt(mean_squared_error(all_l_te, all_p_te)), mean_absolute_error(all_l_te, all_p_te)
    
    # Dọn dẹp RAM/VRAM
    del ensemble_models, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    df_res = pd.DataFrame(fold_results)
    return df_res['r2'].mean(), df_res['rmse'].mean(), df_res['mae'].mean(), ens_r2, ens_rmse, ens_mae

# ================================================================
# MAIN ABLATION SCRIPT
# ================================================================
def main():
    cfg = BaseConfig()
    seed_everything(cfg.SEED)

    # SETUP LOGGING
    os.makedirs("ablation_results", exist_ok=True)
    log_file_path = os.path.join("ablation_results", f"Ablation_Log_{cfg.RUN_TIMESTAMP}.txt")
    
    logger = logging.getLogger('Ablation')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter('%(message)s')
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info("⏳ Đang tải và xử lý dữ liệu gốc...")
    full_dataset = PainDataset(cfg)
    
    # --- CHIA DATA DUY NHẤT 1 LẦN (SUBJECT SPLIT) ---
    all_indices = np.arange(len(full_dataset))
    all_labels  = np.array(full_dataset.stratify_labels)
    all_groups  = np.array(full_dataset.groups)

    counts = Counter(all_labels)
    rare_groups = [g for g, c in counts.items() if c < 2]
    if rare_groups:
        mask = np.isin(all_labels, rare_groups, invert=True)
        all_indices, all_labels, all_groups = all_indices[mask], all_labels[mask], all_groups[mask]

    subject_to_race = {full_dataset.groups[i]: full_dataset.data_list[i]['race'] for i in all_indices}
    asian_subjects = [s for s, r in subject_to_race.items() if r == 'Asian']
    black_subjects = [s for s, r in subject_to_race.items() if r == 'Black']
    white_subjects = [s for s, r in subject_to_race.items() if r == 'White']

    import random
    rng = random.Random(cfg.SEED)
    rng.shuffle(asian_subjects); rng.shuffle(black_subjects); rng.shuffle(white_subjects)
    
    test_subjects = set(asian_subjects[:6] + black_subjects[:12] + white_subjects[:18])
    train_val_idx = np.array([i for i in all_indices if full_dataset.groups[i] not in test_subjects])
    test_idx      = np.array([i for i in all_indices if full_dataset.groups[i] in test_subjects])

    sgkf = StratifiedGroupKFold(n_splits=cfg.K_FOLDS, shuffle=True, random_state=cfg.SEED)
    y_tv = [full_dataset.stratify_labels[i] for i in train_val_idx]
    g_tv = [full_dataset.groups[i] for i in train_val_idx]
    kfold_splits = list(sgkf.split(train_val_idx, y_tv, groups=g_tv))

    logger.info(f"✅ Dữ liệu hoàn tất! Train/Val: {len(train_val_idx)} ảnh | Test: {len(test_idx)} ảnh")
    logger.info(f"🔄 Bắt đầu Ablation Study với {len(EXPERIMENTS)} kịch bản (Lưu log tại {log_file_path})...\n")

    final_results = []
    
    for exp in EXPERIMENTS:
        m_r2, m_rmse, m_mae, e_r2, e_rmse, e_mae = run_single_experiment(
            exp, cfg, full_dataset, train_val_idx, test_idx, kfold_splits, logger, "ablation_results"
        )
        final_results.append({
            "Experiment": exp["exp_id"],
            "Val_R2": round(m_r2, 4),
            "Val_RMSE": round(m_rmse, 4),
            "Test_Ens_R2": round(e_r2, 4),
            "Test_Ens_RMSE": round(e_rmse, 4),
            "Test_Ens_MAE": round(e_mae, 4)
        })

    # XUẤT BẢNG TỔNG KẾT
    df_final = pd.DataFrame(final_results)
    csv_path = os.path.join("ablation_results", "Ablation_Summary_Table.csv")
    txt_path = os.path.join("ablation_results", "Ablation_Summary_Table.txt")
    
    df_final.to_csv(csv_path, index=False)
    
    summary_text = "\n" + "="*80 + "\n"
    summary_text += "🏆 KẾT QUẢ TỔNG HỢP ABLATION STUDY 🏆\n"
    summary_text += "="*80 + "\n"
    summary_text += df_final.to_markdown(index=False)
    summary_text += "\n" + "="*80 + "\n"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    logger.info(summary_text)
    logger.info(f"✅ Hoàn tất! Bảng kết quả đã lưu tại:")
    logger.info(f"   - {csv_path}")
    logger.info(f"   - {txt_path}")
    logger.info(f"   - File log chi tiết: {log_file_path}")

if __name__ == '__main__':
    main()