# -- coding: utf-8 --
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import timm
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ================================================================
# 1. ĐIỀN TÊN THƯ MỤC CHỨA CHECKPOINT CỦA BẠN VÀO ĐÂY
# Ví dụ: "2026-04-12_18-13-57_GRAY_PureRaw_Baseline_fold_1"
# ================================================================
TRAINED_TIMESTAMP = "2026-04-15_11-21-45_GRAY_PureRaw_Baseline" 


class Config:
    DATA_ROOT_DIR = "dataset_osfstorage-archive/Stimuli"
    LABELS_PATH = "dataset_osfstorage-archive/NormingData/DFD_NormingData.csv"
    SKIN_COLOR_BASE = os.path.join("dataset_osfstorage-archive", "Color Skin")

    BATCH_SIZE = 16
    IMG_SIZE = 300
    K_FOLDS = 10
    TEST_SPLIT = 0.15

    CONFIDENCE_THRESHOLD = 0.8
    BELIEVABILITY_THRESHOLD = 2.5
    BACKBONE = 'efficientnet_b3'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    NUM_WORKERS = 4

cfg = Config()

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return torch.stack([x[0] for x in batch]), torch.stack([x[1] for x in batch]), [x[2] for x in batch]

# --- GIỮ NGUYÊN CÁC CLASS DATASET & MODEL TỪ GRAY_BASELINE ---
class SmartFaceCLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    def __call__(self, img):
        if not isinstance(img, Image.Image): return img
        img_np = np.array(img.convert("RGB"))
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        lab_adjusted = cv2.merge((self.clahe.apply(l), a, b))
        return Image.fromarray(cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2RGB))

class BaselinePainModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(cfg.BACKBONE, pretrained=False, num_classes=0, global_pool='avg')
        self.pain_head = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(self.backbone.num_features, 1))
    def forward(self, x):
        return self.pain_head(self.backbone(x))

class PainDataset(Dataset):
    def __init__(self, config):
        self.cfg = config
        self.image_path_map = self._create_smart_file_map(self.cfg.DATA_ROOT_DIR)
        self.skin_color_map = self._create_skin_color_map(self.cfg.SKIN_COLOR_BASE)
        df = pd.read_csv(self.cfg.LABELS_PATH)
        df.columns = df.columns.str.strip()
        for col in ['Pain_Expression', 'PhysicalPain_Neutral', 'HowBelievable', 'OpenFace_confidence']:
            if col in df.columns: df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        if 'OpenFace_confidence' in df.columns: df = df[df['OpenFace_confidence'] >= self.cfg.CONFIDENCE_THRESHOLD]
        if 'HowBelievable' in df.columns: df = df[df['HowBelievable'].fillna(9.0) >= self.cfg.BELIEVABILITY_THRESHOLD]

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
        return {os.path.splitext(f)[0].lower().replace('_cropped', '').replace('_standardized', '').replace('_unedited', ''): os.path.join(dp, f) 
                for dp, _, fs in os.walk(root_dir) for f in fs if f.lower().endswith(('.jpg', '.png'))}
    def _create_skin_color_map(self, skin_base):
        return {os.path.splitext(f)[0].lower().replace('_cropped', '').replace('_standardized', '').replace('_unedited', ''): label 
                for folder, label in {'black': 'Black', 'white': 'White', 'yellow': 'Asian'}.items() 
                if os.path.exists(os.path.join(skin_base, folder)) 
                for f in os.listdir(os.path.join(skin_base, folder)) if f.lower().endswith(('.jpg', '.png'))}
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        item = self.data_list[idx]
        return ImageOps.exif_transpose(Image.open(item['path'])).convert("RGB"), torch.tensor(item['label'], dtype=torch.float), item['group']

class SplitTransformDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base, self.transform = base_dataset, transform
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        item = self.base.data_list[idx]
        img = ImageOps.exif_transpose(Image.open(item['path'])).convert("RGB")
        return self.transform(img) if self.transform else img, torch.tensor(item['label'], dtype=torch.float), item['group']


# ================================================================
# HÀM PHÂN TÍCH FAIRNESS CHUYÊN SÂU
# ================================================================
def analyze_fairness_deeply(df, export_dir=None):
    if 'Error' not in df.columns:
        df['Error'] = np.abs(df['TrueScore'] - df['PredScore'])
        
    df['IndividualBias'] = df['PredScore'] - df['TrueScore']
    
    # ---------------------------------------------------------
    # BƯỚC LỌC THÔNG MINH ĐỂ VẼ BIỂU ĐỒ (Chỉ vẽ nhóm >= 2 mẫu)
    # ---------------------------------------------------------
    counts = df['ColorGroup'].value_counts()
    valid_groups = counts[counts >= 2].index.tolist() # Chỉ lấy nhóm có >= 2 mẫu
    
    # DataFrame dành riêng cho việc vẽ hình
    df_plot = df[df['ColorGroup'].isin(valid_groups)]
    
    print("\n" + "="*80)
    print("📊 THỐNG KÊ SỐ LƯỢNG MẪU TRONG TẬP TEST:")
    print("="*80)
    for race, count in counts.items():
        if count < 2:
            print(f"   - {race}: {count} ảnh (⚠️ Bị loại khỏi biểu đồ do quá ít dữ liệu)")
        else:
            print(f"   - {race}: {count} ảnh")
    print("-" * 80)

    # --- 1. Vẽ biểu đồ với df_plot ---
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_plot, x='ColorGroup', y='Error', hue='ColorGroup', palette='Set2', width=0.5, legend=False)
    sns.stripplot(data=df_plot, x='ColorGroup', y='Error', color=".3", alpha=0.4, jitter=True)
    plt.title("SỰ BIẾN THIÊN CỦA SAI SỐ TUYỆT ĐỐI (MAE)", fontsize=13, fontweight='bold')
    plt.ylabel("Absolute Error (|True - Pred|)")
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sns.violinplot(data=df_plot, x='ColorGroup', y='IndividualBias', hue='ColorGroup', inner="quartile", palette='Pastel1', legend=False)
    plt.axhline(0, color='red', linestyle='--', alpha=0.6, label='Zero Bias')
    plt.title("PHÂN BỔ ĐỘ LỆCH (BIAS) DỰ ĐOÁN", fontsize=13, fontweight='bold')
    plt.ylabel("Bias (Predicted - Actual Score)")
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        plt.savefig(os.path.join(export_dir, "fairness_analysis_plot.png"), dpi=300)
        print(f"📸 Đã lưu biểu đồ Fairness tại: {os.path.join(export_dir, 'fairness_analysis_plot.png')}")
    plt.show()

    # --- 2. Kiểm định thống kê ---
    print("\n" + "="*80)
    print(" 🔬 KIỂM TRA TÍNH CÓ Ý NGHĨA THỐNG KÊ (P-VALUE):")
    print("="*80)
    
    # Cũng chỉ chạy T-test cho các nhóm hợp lệ
    for i in range(len(valid_groups)):
        for j in range(i + 1, len(valid_groups)):
            g1_err = df_plot[df_plot['ColorGroup'] == valid_groups[i]]['Error']
            g2_err = df_plot[df_plot['ColorGroup'] == valid_groups[j]]['Error']
            
            t_stat, p_val = stats.ttest_ind(g1_err, g2_err, equal_var=False)
            
            if np.isnan(p_val):
                print(f"- So sánh {valid_groups[i]:<6} vs {valid_groups[j]:<6}: LỖI PHÂN BỐ")
            else:
                status = "🔴 CÓ KHÁC BIỆT Ý NGHĨA (Bias)" if p_val < 0.05 else "🟢 KHÔNG CÓ KHÁC BIỆT (Fair)"
                print(f"- So sánh {valid_groups[i]:<6} vs {valid_groups[j]:<6}: p-value = {p_val:.4f} -> {status}")
    print("="*80)

# ================================================================
# CHẠY TEST TRỰC TIẾP
# ================================================================
def main(config):
    seed_everything(config.SEED)
    print("🚀 ĐANG TÁI TẠO LẠI TẬP DỮ LIỆU ĐỂ LẤY TẬP TEST GỐC...")
    
    eval_transform = transforms.Compose([
        SmartFaceCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.449, 0.449, 0.449], std=[0.226, 0.226, 0.226])
    ])

    full_dataset = PainDataset(config)
    eval_dataset = SplitTransformDataset(full_dataset, eval_transform)

    all_indices = np.arange(len(full_dataset))
    all_labels = np.array(full_dataset.stratify_labels)
    all_groups = np.array(full_dataset.groups)
    
    # Lọc rare groups giống hệt lúc train
    from collections import Counter
    counts = Counter(all_labels)
    rare_groups = [g for g, c in counts.items() if c < 2]
    if rare_groups:
        valid_mask = np.isin(all_labels, rare_groups, invert=True)
        all_indices, all_labels, all_groups = all_indices[valid_mask], all_labels[valid_mask], all_groups[valid_mask]

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

    test_idx_list = []
    for idx in all_indices:
        if full_dataset.groups[idx] in test_subjects:
            test_idx_list.append(idx)

    test_idx = np.array(test_idx_list)
    # =========================================================================

    import sys
    test_loader = DataLoader(
        Subset(eval_dataset, test_idx), batch_size=config.BATCH_SIZE, shuffle=False, 
        collate_fn=custom_collate, num_workers=0 if sys.platform == "win32" else config.NUM_WORKERS
    )

    print(f"✅ Đã tạo xong tập test: {len(test_idx)} ảnh.")
    print("⏳ ĐANG LOAD CÁC MODEL CHECKPOINTS ĐÃ TRAIN...")

    ensemble_models = []
    for i in range(config.K_FOLDS):
        cp_path = os.path.join("checkpoints", f"{TRAINED_TIMESTAMP}_fold_{i+1}.pth")
        if os.path.exists(cp_path):
            m = BaselinePainModel(config).to(config.DEVICE)
            m.load_state_dict(torch.load(cp_path, map_location=config.DEVICE, weights_only=True))
            m.eval()
            ensemble_models.append(m)
        else:
            print(f"⚠️ [Cảnh báo] Không tìm thấy: {cp_path}")

    if len(ensemble_models) == 0:
        print("❌ LỖI: KHÔNG TÌM THẤY CHECKPOINT NÀO. Vui lòng kiểm tra lại biến TRAINED_TIMESTAMP ở dòng 19.")
        return

    print(f"✅ Load thành công {len(ensemble_models)}/{config.K_FOLDS} models. Đang chạy dự đoán...")

    all_labels_test, all_preds_test, all_races_test = [], [], []
    with torch.no_grad():
        for batch_data in test_loader:
            if batch_data is None: continue
            imgs, labels, batch_groups = batch_data
            imgs = imgs.to(config.DEVICE)
            
            fold_preds = [m(imgs).squeeze(-1).cpu().numpy() for m in ensemble_models]
            ensemble_pred = np.mean(fold_preds, axis=0)
            
            all_labels_test.extend(labels.numpy())
            all_preds_test.extend(ensemble_pred.tolist())
            
            # Lấy thông tin màu da thực tế từ dataset thay vì split() từ group
            # Vì group hiện tại đã được dùng để lưu Subject ID
            batch_races = []
            for i in range(len(batch_groups)):
                # Tìm race bằng cách lấy thông qua mapping của dataset
                # Trong custom_collate chúng ta chỉ có group_id (giờ là Subject ID), nên cần truy xuất ngược
                pass
            
            # Cách an toàn nhất: lấy race thông qua dataset
            
    # Lấy lại race trực tiếp bằng test_idx để chính xác hơn (vì batch_groups hiện lưu Subject ID)
    for idx in test_idx:
        all_races_test.append(full_dataset.data_list[idx]['race'])

    # Đánh giá cơ bản
    test_r2 = r2_score(all_labels_test, all_preds_test)
    test_rmse = np.sqrt(mean_squared_error(all_labels_test, all_preds_test))
    test_mae = mean_absolute_error(all_labels_test, all_preds_test)
    print(f"\n📊 KẾT QUẢ ENSEMBLE TEST | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R²: {test_r2:.4f}")

    # Chạy phân tích Fairness
    df_test = pd.DataFrame({
        'TrueScore': all_labels_test,
        'PredScore': all_preds_test,
        'ColorGroup': all_races_test
    })
    
    export_dir = os.path.join("exported_data", f"{TRAINED_TIMESTAMP}_EVALUATION")
    analyze_fairness_deeply(df_test, export_dir)

if __name__ == '__main__':
    main(cfg)