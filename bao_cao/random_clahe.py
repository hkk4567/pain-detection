import os
import cv2
import random
import glob
import numpy as np
import matplotlib.pyplot as plt

class SmartFaceCLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def process(self, image_path):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Chuyển sang không gian LAB
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Chỉ áp dụng CLAHE trên kênh L
        l_enhanced = self.clahe.apply(l)
        
        # Ghép lại
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        img_enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        return img_rgb, img_enhanced_rgb

def visualize_comparison(original, enhanced, save_path="hinh_bu_sang.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original)
    axes[0].set_title("Original Image (Before)")
    axes[0].axis("off")
    
    axes[1].imshow(enhanced)
    axes[1].set_title("Smart Face CLAHE (After)")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Đã lưu hình ảnh so sánh tại: {save_path}")

def get_random_image_from_dataset(root_dir):
    """Quét toàn bộ thư mục và trả về đường dẫn của 1 ảnh ngẫu nhiên"""
    print(f"Đang tìm kiếm ảnh trong: {root_dir}...")
    image_paths = []
    
    # Tìm tất cả file .jpg, .jpeg, .png trong các thư mục con
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
    
    if not image_paths:
        print("❌ Không tìm thấy ảnh nào trong thư mục!")
        return None
        
    random_image = random.choice(image_paths)
    return random_image

# ==========================================
# CHẠY THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    processor = SmartFaceCLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
    
    # Trỏ đến thư mục chứa ảnh gốc theo cấu trúc project của bạn
    DATASET_DIR = "dataset_osfstorage-archive/Stimuli" 
    
    # Lấy 1 ảnh ngẫu nhiên
    random_img_path = get_random_image_from_dataset(DATASET_DIR)
    
    if random_img_path:
        print(f"📸 Đã chọn ngẫu nhiên ảnh: {random_img_path}")
        result = processor.process(random_img_path)
        
        if result is not None:
            original, enhanced = result
            # Đặt tên file đầu ra dựa trên tên ảnh gốc để dễ theo dõi
            base_name = os.path.basename(random_img_path)
            output_filename = f"so_sanh_CLAHE_{base_name}"
            
            visualize_comparison(original, enhanced, save_path=output_filename)
        else:
            print("❌ Lỗi khi đọc ảnh bằng OpenCV.")