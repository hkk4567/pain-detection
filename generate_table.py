import pandas as pd
import os

# 1. Đọc lại file kết quả đã lưu
csv_path = os.path.join("ablation_results", "Ablation_Summary_Table.csv")
txt_path = os.path.join("ablation_results", "Ablation_Summary_Table.txt")

try:
    df_final = pd.read_csv(csv_path)

    # 2. Tạo text tổng kết
    summary_text = "\n" + "="*80 + "\n"
    summary_text += "🏆 KẾT QUẢ TỔNG HỢP ABLATION STUDY 🏆\n"
    summary_text += "="*80 + "\n"
    summary_text += df_final.to_markdown(index=False)
    summary_text += "\n" + "="*80 + "\n"

    # 3. Ghi ra file txt và in ra màn hình
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    print("✅ Đã khôi phục và tạo file bảng Text thành công!")

except FileNotFoundError:
    print("❌ Không tìm thấy file CSV, bạn kiểm tra lại đường dẫn nhé.")