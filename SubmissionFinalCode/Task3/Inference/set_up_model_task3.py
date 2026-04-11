import gdown
import os
import zipfile
from Config import config

Task_3_Train_Test_Config = config.return_Task3_Train_Test_Config()
save_path = Task_3_Train_Test_Config["weight"]

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
        print(f"✅ Dữ liệu Task 3 đã tồn tại trong '{output_dir}'. Bỏ qua bước tải!")
        return

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, 'weights_task3.zip')
    url = f'https://drive.google.com/uc?id={folder_id}'

    try:
        print(f"⬇️ Đang tải dữ liệu Task 3 từ Drive...")
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
    task_3_file_id = '10B_e4lGqCWoAEQcaGSKshkcFVY1wpMi1' 
    download_task_weights(task_3_file_id, save_path)