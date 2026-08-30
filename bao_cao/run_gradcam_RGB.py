# -- coding: utf-8 --
import os
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
import timm

# Khai báo thư viện Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ================================================================
# 1. ĐIỀN TÊN THƯ MỤC CHỨA CHECKPOINT RGB VÀ ẢNH MUỐN XEM
# ================================================================
# VÍ DỤ: "2026-04-15_17-07-38_V11_RGB_Fairness"
# 2026-04-15_17-07-38_V11_RGB_Fairness_fold_1.pth
TRAINED_TIMESTAMP = "2026-04-15_17-07-38_V11_RGB_Fairness"
TARGET_IMAGE_NAME = "dpd_1_of3_p2b" # Tên ảnh cần lấy (chữ thường, không đuôi mở rộng)

# ================================================================
# CONFIG & MODEL (GIỮ NGUYÊN TỪ SCRIPT RGB)
# ================================================================
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

        self.data_list = []
        for _, row in df.iterrows():
            target = str(row['Target']).strip()
            clean_name = os.path.splitext(target)[0].lower().replace('_cropped', '').replace('_standardized', '').replace('_unedited', '')
            actual_key = clean_name if clean_name in self.image_path_map else (clean_name + "_earring" if (clean_name + "_earring") in self.image_path_map else None)
            if actual_key is None or actual_key not in self.skin_color_map: continue
            
            p_val, n_val = row['Pain_Expression'], row['PhysicalPain_Neutral']
            label = float(p_val) if not pd.isna(p_val) else float(n_val) if not pd.isna(n_val) else None
            if label is not None:
                self.data_list.append({'path': self.image_path_map[actual_key], 'label': label, 'name': actual_key})

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
        img = ImageOps.exif_transpose(Image.open(item['path'])).convert("RGB")
        return img, item['label'], item['name']


# ================================================================
# HÀM UN-NORMALIZE ĐỂ VẼ ẢNH GỐC (DÀNH CHO RGB)
# ================================================================
def unnormalize_and_to_rgb(tensor_img):
    """Đảo ngược Normalize để lấy lại ảnh gốc phục vụ vẽ Heatmap"""
    # CHÚ Ý: Đã đổi sang mean/std của hệ màu RGB
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_np = tensor_img.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    return img_np

# ================================================================
# CHƯƠNG TRÌNH CHÍNH
# ================================================================
def main():
    print("🚀 BẮT ĐẦU CHẠY ENSEMBLE GRAD-CAM (MÔ HÌNH RGB)...")
    
    # 1. Load TOÀN BỘ 10 Model (Ensemble)
    ensemble_models = []
    ensemble_cams = [] # Lưu các bộ tạo CAM cho từng model
    
    for i in range(cfg.K_FOLDS):
        cp_path = os.path.join("checkpoints", f"{TRAINED_TIMESTAMP}_fold_{i+1}.pth")
        if os.path.exists(cp_path):
            m = BaselinePainModel(cfg).to(cfg.DEVICE)
            m.load_state_dict(torch.load(cp_path, map_location=cfg.DEVICE, weights_only=True))
            m.eval()
            ensemble_models.append(m)
            
            # Khởi tạo GradCAM cho model này (lớp conv_head của EfficientNet)
            cam = GradCAM(model=m, target_layers=[m.backbone.conv_head])
            ensemble_cams.append(cam)
        else:
            print(f"⚠️ Không tìm thấy Fold {i+1}")

    if len(ensemble_models) == 0:
        print("❌ LỖI: Không load được bất kỳ model nào!")
        return
        
    print(f"✅ Đã load thành công {len(ensemble_models)}/{cfg.K_FOLDS} Models để làm Ensemble.")

    # Target là node số 0 (Dự đoán Pain score)
    targets = [ClassifierOutputTarget(0)]

    # 2. Load Dữ liệu
    # CHÚ Ý: Đã bỏ transforms.Grayscale và đổi Normalize sang chuẩn RGB
    eval_transform = transforms.Compose([
        SmartFaceCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = PainDataset(cfg)
    
    target_idx = -1
    for idx, item in enumerate(dataset.data_list):
        if TARGET_IMAGE_NAME in item['name']:
            target_idx = idx
            break
            
    if target_idx == -1:
        print(f"❌ LỖI: Không tìm thấy ảnh có tên chứa '{TARGET_IMAGE_NAME}' trong dataset thỏa mãn điều kiện lọc.")
        return
        
    indices_to_show = [target_idx]
    
    export_dir = os.path.join("exported_data", f"{TRAINED_TIMESTAMP}_ENSEMBLE_GRADCAM")
    os.makedirs(export_dir, exist_ok=True)
    
    print(f"⏳ Đang xử lý ảnh {TARGET_IMAGE_NAME} với Ensemble Grad-CAM...")
    
    for i, idx in enumerate(indices_to_show):
        pil_img, true_label, img_name = dataset[idx]
        
        # Tiền xử lý
        input_tensor = eval_transform(pil_img).unsqueeze(0).to(cfg.DEVICE)
        
        # Các biến để cộng dồn (Ensemble)
        sum_preds = 0.0
        sum_heatmaps = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE), dtype=np.float32)
        
        # Chạy qua từng Model trong Ensemble
        for m, cam in zip(ensemble_models, ensemble_cams):
            # Dự đoán
            with torch.no_grad():
                pred = m(input_tensor).item()
                sum_preds += pred
            
            # Tính Heatmap
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            sum_heatmaps += grayscale_cam
            
        # Chia trung bình để lấy kết quả Ensemble
        ensemble_pred = sum_preds / len(ensemble_models)
        ensemble_heatmap = sum_heatmaps / len(ensemble_models)
        
        # TẠO MẶT NẠ XÓA LỖI ĐỐM SÁNG Ở CÁC GÓC/VIỀN
        h, w = ensemble_heatmap.shape
        mask = np.zeros((h, w), dtype=np.float32)
        # Giữ lại vùng giữa (từ 10% đến 90% kích thước ảnh)
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        cv2.rectangle(mask, (margin_x, margin_y), (w - margin_x, h - margin_y), 1.0, -1)
        # Làm mờ viền mặt nạ bằng GaussianBlur (Kernel size 51)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)

        # Nhân heatmap với mặt nạ để dập tắt viền
        ensemble_heatmap = ensemble_heatmap * mask
        
        # CHUẨN HÓA LẠI: scale phần trung tâm lên lại khoảng [0, 1] để mặt có màu đỏ rực
        ensemble_heatmap = (ensemble_heatmap - np.min(ensemble_heatmap)) / (np.max(ensemble_heatmap) - np.min(ensemble_heatmap) + 1e-8)
        
        # Lấy ảnh gốc để đè heatmap lên
        rgb_img = unnormalize_and_to_rgb(input_tensor)
        
        # Trộn ảnh gốc và heatmap (dùng thư viện)
        cam_image = show_cam_on_image(rgb_img, ensemble_heatmap, use_rgb=True)
        
        # Vẽ biểu đồ
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Original Image\nTrue Pain: {true_label:.2f}")
        axes[0].axis('off')
        
        im = axes[1].imshow(ensemble_heatmap, cmap='jet')
        axes[1].set_title("Ensemble Grad-CAM\n(Averaged 10 Folds)")
        axes[1].axis('off')
        
        axes[2].imshow(cam_image)
        axes[2].set_title(f"Overlay\nEnsemble Pred Pain: {ensemble_pred:.2f}")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        save_path = os.path.join(export_dir, f"ensemble_cam_RGB_{img_name}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  -> Đã lưu {save_path}")

    print(f"\n🎉 HOÀN TẤT! Ảnh ENSEMBLE Grad-CAM (RGB) đã được lưu vào thư mục:\n{export_dir}")

if __name__ == '__main__':
    main()