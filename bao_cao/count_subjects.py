# -- coding: utf-8 --
import os
import pandas as pd
from collections import defaultdict

# ================================================================
# CẤU HÌNH ĐƯỜNG DẪN VÀ NGƯỠNG LỌC (Giống trong Config)
# ================================================================
LABELS_PATH = "dataset_osfstorage-archive/NormingData/DFD_NormingData.csv"
SKIN_COLOR_BASE = os.path.join("dataset_osfstorage-archive", "Color Skin")

CONFIDENCE_THRESHOLD = 0.8
BELIEVABILITY_THRESHOLD = 2.5

def create_skin_color_map(skin_base):
    """Hàm tạo từ điển map từ tên file sang nhãn màu da"""
    color_map = {}
    if not os.path.exists(skin_base):
        print(f"❌ Không tìm thấy thư mục: {skin_base}")
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

def main():
    print("⏳ Đang đọc và xử lý dữ liệu...")
    skin_color_map = create_skin_color_map(SKIN_COLOR_BASE)

    if not os.path.exists(LABELS_PATH):
        print(f"❌ Không tìm thấy file nhãn: {LABELS_PATH}")
        return

    df = pd.read_csv(LABELS_PATH)
    df.columns = df.columns.str.strip()

    # Chuyển đổi các cột sang dạng số
    cols_to_numeric = ['Pain_Expression', 'PhysicalPain_Neutral', 'HowBelievable', 'OpenFace_confidence']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

    # Lọc dữ liệu theo threshold
    original_len = len(df)
    if 'OpenFace_confidence' in df.columns:
        df = df[df['OpenFace_confidence'] >= CONFIDENCE_THRESHOLD]
    if 'HowBelievable' in df.columns:
        df = df[df['HowBelievable'].fillna(9.0) >= BELIEVABILITY_THRESHOLD]

    print(f"✅ Đã lọc {original_len - len(df)} mẫu dưới ngưỡng threshold.")

    # Dictionary để lưu danh sách Subject IDs (dùng set để tự động loại bỏ trùng lặp)
    subjects_by_race = defaultdict(set)
    # Dictionary để đếm số lượng ảnh
    samples_by_race = defaultdict(int)

    # Biến để log số ảnh bị loại vì các lý do khác
    missing_skin_count = 0
    missing_label_count = 0

    for _, row in df.iterrows():
        target = str(row['Target']).strip()
        base_name = os.path.splitext(target)[0].lower()
        clean_name = base_name.replace('_cropped', '').replace('_standardized', '').replace('_unedited', '')
        
        # 1. Trích xuất Subject ID (Logic từ gray_baseline.py)
        parts = target.split('_')
        if len(parts) >= 3:
            subject_id = parts[2].upper() # Ví dụ: AF6
        else:
            # [SỬA LỖI ĐỒNG BỘ] Giữ nguyên chữ thường giống file gray_baseline.py để tránh lệch hash của Set
            subject_id = clean_name 

        # 2. Map Skin Color
        actual_key = None
        if clean_name in skin_color_map:
            actual_key = clean_name
        elif (clean_name + "_earring") in skin_color_map:
            actual_key = clean_name + "_earring"
            
        if actual_key is None:
            missing_skin_count += 1
            continue
            
        # 3. Check xem có label không (Pain_Expression hoặc PhysicalPain_Neutral)
        p_val = row['Pain_Expression']
        n_val = row['PhysicalPain_Neutral']
        if pd.isna(p_val) and pd.isna(n_val):
            missing_label_count += 1
            continue
            
        # 4. Ghi nhận dữ liệu hợp lệ
        race = skin_color_map[actual_key]
        subjects_by_race[race].add(subject_id)
        samples_by_race[race] += 1

    # In Báo cáo thống kê
    print("\n" + "="*65)
    print("📊 BẢNG THỐNG KÊ CHỦ THỂ & SỐ ẢNH THEO MÀU DA")
    print("="*65)
    print(f"{'Màu da (Race)':<15} | {'Số chủ thể (Subjects)':<20} | {'Số lượng ảnh (Samples)':<22}")
    print("-" * 65)
    
    total_samples = 0
    
    # In theo thứ tự chữ cái của nhóm màu da
    for race in sorted(subjects_by_race.keys()):
        n_subjects = len(subjects_by_race[race])
        n_samples = samples_by_race[race]
        
        total_samples += n_samples
        
        print(f"{race:<15} | {n_subjects:<20} | {n_samples:<22}")
    asian_sub = subjects_by_race.get('Asian', set())
    black_sub = subjects_by_race.get('Black', set())
    white_sub = subjects_by_race.get('White', set())

    overlap_AW = asian_sub.intersection(white_sub)
    overlap_AB = asian_sub.intersection(black_sub)
    overlap_BW = black_sub.intersection(white_sub)
        
    # [SỬA LỖI TÍNH TỔNG] Gộp toàn bộ subjects lại thành một tập hợp duy nhất để đếm số lượng thực tế (226)
    all_unique_subjects = set()
    for subjects in subjects_by_race.values():
        all_unique_subjects.update(subjects)
        
    real_total_subjects = len(all_unique_subjects)

    print("-" * 65)
    print(f"{'TỔNG CỘNG':<15} | {real_total_subjects:<20} | {total_samples:<22}")
    print("="*65)
    print(f"ℹ️ Ảnh bị loại do thiếu label (NaN): {missing_label_count}")
    print(f"ℹ️ Ảnh bị loại do không map được màu da: {missing_skin_count}")

    print("\n⚠️ DANH SÁCH CÁC SUBJECT ID BỊ TRÙNG GIỮA CÁC NHÓM MÀU DA (Gây ra chênh lệch tổng số):")
    if overlap_AW: print(f"- Trùng giữa Asian và White ({len(overlap_AW)} ID): {overlap_AW}")
    if overlap_AB: print(f"- Trùng giữa Asian và Black ({len(overlap_AB)} ID): {overlap_AB}")
    if overlap_BW: print(f"- Trùng giữa Black và White ({len(overlap_BW)} ID): {overlap_BW}")
if __name__ == '__main__':
    main()