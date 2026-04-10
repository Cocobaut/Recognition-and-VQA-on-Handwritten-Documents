import os
import zipfile
import subprocess
import sys
from Config import config

Task_5_Predict_Config = config.return_Task5_Predict_Config()

# Cấu hình thông tin
file_id = "1d9SOH4o3p-sgCZqtEBluFxCKhnB_3uyL"  
zip_file_name = path = os.path.join(Task_5_Predict_Config["weight"], "model.zip")
extract_folder = path = os.path.join(Task_5_Predict_Config["weight"], "model")

def install_gdown():
    try:
        import gdown
    except ImportError:
        print("- Đang cài đặt công cụ tải file tự động (gdown)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

def main():
    install_gdown()
    import gdown
    
    print("\n" + "="*60)
    print("- Bắt đầu tải mô hình từ google drive...")
    print("="*60)

    # 1. Tải file
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f"- Đang tải file....")
    gdown.download(url, zip_file_name, quiet=False)

    # 2. Giải nén
    if os.path.exists(zip_file_name):
        print(f"\n- Đang giải nén vào thư mục: {extract_folder}...")
        os.makedirs(extract_folder, exist_ok=True)
        
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

        # 3. Dọn dẹp thùng rác
        os.remove(zip_file_name)
        
        print("- Hoàn tất! Mô hình đã sẵn sàng.")
    else:
        print("\n- Tải file thất bại! Hãy kiểm tra lại link Google Drive.")

if __name__ == "__main__":
    main()