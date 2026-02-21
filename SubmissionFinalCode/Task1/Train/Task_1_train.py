import torch
from ultralytics import YOLO
import Config.config as config

Task_1_Train_Test_Config = config.return_Task1_Train_Test_Config()

def train_process():
    # 0 Kiểm tra GPU
    if torch.cuda.is_available():
        print(f"Đang sử dụng GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CẢNH BÁO: Không tìm thấy GPU, hệ thống sẽ chạy bằng CPU (rất chậm)!")

    # 1. Khởi tạo mô hình YOLO11x - Bản mạnh nhất cho độ chính xác tối ưu
    model = YOLO("yolo11l.pt") 

    # 2. Cấu hình Training chuyên sâu cho bài toán OCR
    results = model.train(
        data="SubmissionFinalCode/Task1/Train/Task_1_set_up_yaml.yaml",
        epochs=30,                  # Train 300 epoch để mô hình hội tụ tốt nhất
        imgsz=800,                  # Tăng độ phân giải lên 1280px để thấy rõ chữ nhỏ
        batch=2,                    # Auto-batch để tận dụng tối đa VRAM GPU
        workers=2,
        device=0,                    # Chạy trên GPU 0
        
        # --- Tham số tối ưu độ chính xác ---
        optimizer='AdamW',           # Optimizer hiện đại, ổn định cho chữ viết tay
        lr0=0.001,                   # Learning rate cơ bản
        cos_lr=True,                 # Sử dụng Cosine annealing để giảm LR mịn màng
        patience=50,                 # Early stopping nếu 50 epoch không cải thiện mAP
        
        # --- Augmentation nặng (Cực kỳ quan trọng cho chữ viết tay) ---
        mosaic=1.0,                  # Trộn ảnh để học vật thể nhỏ ở mọi góc độ
        mixup=0.1,                  # Chồng hình ảnh để tăng khả năng tổng quát hóa
        copy_paste=0.3,              # Sao chép các block chữ để làm giàu dữ liệu
        degrees=0.5,                 # Xoay nhẹ ảnh
        shear=0.1,                   # Làm biến dạng nhẹ nét chữ
        
        # Lưu và quản lý
        project=Task_1_Train_Test_Config["weight"],
        name="YOLO11l_Task1_Precision",
        save=True,
        exist_ok=True
    )

if __name__ == '__main__':
    train_process()