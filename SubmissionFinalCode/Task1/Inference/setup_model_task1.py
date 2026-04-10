import gdown
import os
import Config.config as config

Task_1_Train_Test_Config = config.return_Task1_Train_Test_Config()
weight_path = Task_1_Train_Test_Config["weight"]

def download_weights_from_drive(output_dir):
    folder_id = '1Bkx3qlKuRqSHNCls0_iaBcFjz4F5XQeM'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục: {output_dir}")

    try:
        print("Đang bắt đầu tải trọng số...")
        gdown.download_folder(id=folder_id, output=output_dir, quiet=False, use_cookies=False)
        print(f"\nThành công! Trọng số đã được lưu tại: {output_dir}")
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    download_weights_from_drive(weight_path)