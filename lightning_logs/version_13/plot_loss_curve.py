import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập giao diện biểu đồ chuyên nghiệp
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11, 
    'axes.labelsize': 12, 
    'axes.titlesize': 14,
    'figure.titlesize': 16
})

# 1. Định vị file metrics.csv nằm cùng thư mục với script này
csv_path = 'metrics.csv'

if not os.path.exists(csv_path):
    print(f"[ERROR] Không tìm thấy file {csv_path} ở thư mục hiện tại.")
    print("Hãy chắc chắn bạn đang đứng ở thư mục 'lightning_logs\\version_13' khi chạy script.")
    exit()

# 2. Đọc dữ liệu thực tế từ log của bạn
print("[INFO] Đang đọc dữ liệu từ metrics.csv...")
df = pd.read_csv(csv_path)

# Lọc các dòng hợp lệ cho Train và Validation (tránh các dòng trống NaN do Lightning tạo ra)
train_df = df[['step', 'train_loss_step']].dropna()
val_df = df[['step', 'val_loss']].dropna()

if train_df.empty:
    print("[WARN] Cột 'train_loss_step' không có dữ liệu hợp lệ.")
if val_df.empty:
    print("[WARN] Cột 'val_loss' trống. Mô hình có thể chưa chạy hết Epoch nào để tính Validation.")

# 3. Tiến hành vẽ đồ thị
plt.figure(figsize=(10, 6))

# Vẽ đường gốc Train Loss (màu xanh nhạt dải nền để thể hiện độ dao động batch)
plt.plot(train_df['step'], train_df['train_loss_step'], 
         color='#4a90e2', alpha=0.2, label='Train Loss (Step)')

# Làm mịn đường Train Loss bằng kỹ thuật Moving Average (cửa sổ 100 steps) để thấy rõ xu hướng hội tụ
train_df['train_loss_smooth'] = train_df['train_loss_step'].rolling(window=100, min_periods=1).mean()
plt.plot(train_df['step'], train_df['train_loss_smooth'], 
         color='#1f77b4', linewidth=2, label='Train Loss (Smoothed)')

# Vẽ các điểm mốc Validation Loss (Đường nối điểm tròn màu cam nổi bật)
if not val_df.empty:
    plt.plot(val_df['step'], val_df['val_loss'], 
         color='#ff7f0e', marker='o', linewidth=2.5, markersize=6, label='Validation Loss')
    # Giới hạn trục Y hợp lý để tránh các điểm vọt đỉnh (outliers) ở step đầu tiên làm hỏng scale đồ thị
    plt.ylim(0, val_df['val_loss'].iloc[0] * 1.5)
else:
    plt.ylim(0, train_df['train_loss_smooth'].max() * 1.2)

# Cấu hình tiêu đề và nhãn trục
plt.title('SSTZIP-GNN Learning Curve (Real Metrics)', fontweight='bold', pad=15)
plt.xlabel('Training Steps', fontweight='bold')
plt.ylabel('Loss Value', fontweight='bold')

plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()

# 4. Lưu đồ thị thành file ảnh chất lượng cao 300 DPI
output_image = 'SSTZIP_GNN_Actual_Loss_Curve.png'
plt.savefig(output_image, dpi=300)
plt.show()

print(f"[OK] Đã xuất thành công biểu đồ loss curve thật tại: {os.path.abspath(output_image)}")
