import gdown
import os
import zipfile
from Config import config

Task_4_Train_Test_Config = config.return_Task4_Train_Test_Config()
save_path = Task_4_Train_Test_Config["weight"]

def check_data_exists(directory):
    if not os.path.exists(directory):
        return False
    for root, dirs, files in os.walk(directory):
        for f in files:
            if not f.startswith('.') and f != "desktop.ini":
                return True
    return False

def download_task_weights(folder_id, output_dir):
    if check_data_exists(output_dir):
        print(f"✅ Dữ liệu Task 4 đã sẵn sàng tại '{output_dir}'. Bỏ qua tải xuống!")
        return

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, 'weights_task4.zip')
    url = f'https://drive.google.com/uc?id={folder_id}'

    try:
        print(f"⬇️ Đang tải dữ liệu Task 4 từ Drive...")
        gdown.download(url, zip_path, quiet=False)
        
        if os.path.exists(zip_path):
            print("📦 Đang giải nén...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(zip_path)
            
        print(f"🎉 Thành công! Dữ liệu đã được lưu tại: {output_dir}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    task_4_file_id = '1AphulOyO5CKxWWZeM0cZ8PkW2vNuv8_e'
    download_task_weights(task_4_file_id, save_path)