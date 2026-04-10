import gdown
import os
from Config import config

Task_4_Train_Test_Config = config.return_Task4_Train_Test_Config()

def download_task_weights(folder_id, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục: {output_dir}")

    try:
        print(f"Đang tải dữ liệu từ Drive (ID: {folder_id})...")
        gdown.download_folder(id=folder_id, output=output_dir, quiet=False, use_cookies=False)
        print(f"Thành công! Dữ liệu đã được lưu tại: {output_dir}")
    except Exception as e:
        print(f"Lỗi: {e}")

task_4_folder_id = '1E-uFIVnDO7RmhCpuE4UhwhyIeWpjmG6I'
save_path = Task_4_Train_Test_Config["weight"]

if __name__ == "__main__":
    download_task_weights(task_4_folder_id, save_path)