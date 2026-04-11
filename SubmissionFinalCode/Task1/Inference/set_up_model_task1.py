import gdown
import os
import zipfile
import Config.config as config

Task_1_Train_Test_Config = config.return_Task1_Train_Test_Config()
weight_path = Task_1_Train_Test_Config["weight"]

def check_data_exists(directory):
    """Kiểm tra xem thư mục có thực sự chứa file dữ liệu không (bỏ qua file rác)"""
    if not os.path.exists(directory):
        return False
    for root, dirs, files in os.walk(directory):
        for f in files:
            if not f.startswith('.') and f != "desktop.ini":
                return True # Thấy có file thật -> Đã tải
    return False

def download_weights_from_drive(output_dir):
    file_id = '18g0UqiTbB6mmoo-xrK8pZ68UR0u3fWzd'
    
    # Dùng hàm check xịn xò
    if check_data_exists(output_dir):
        print(f"✅ Trọng số đã có sẵn tại '{output_dir}'. Bỏ qua bước tải xuống!")
        return

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, 'weights.zip')
    url = f'https://drive.google.com/uc?id={file_id}'

    try:
        print("⬇️ Đang bắt đầu tải trọng số Task 1 (vui lòng đợi)...")
        gdown.download(url, zip_path, quiet=False)
        
        if os.path.exists(zip_path):
            print("📦 Đang giải nén dữ liệu...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(zip_path)
            
        print(f"🎉 Thành công! Trọng số đã sẵn sàng tại: {output_dir}")
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    download_weights_from_drive(weight_path)