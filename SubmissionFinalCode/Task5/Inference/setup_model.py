import os
import zipfile
import subprocess
import sys

# ==========================================
# CẤU HÌNH THÔNG TIN GOOGLE DRIVE
# ==========================================
FILE_ID = "1d9SOH4o3p-sgCZqtEBluFxCKhnB_3uyL"  
ZIP_FILE_NAME = "model.zip"
EXTRACT_FOLDER = "model" 

def install_gdown():
    try:
        import gdown
    except ImportError:
        print("⚙️ Đang cài đặt công cụ tải file tự động (gdown)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

def main():
    install_gdown()
    import gdown
    
    print("\n" + "="*60)
    print("🚀 BẮT ĐẦU TẢI MÔ HÌNH TỪ GOOGLE DRIVE...")
    print("="*60)

    # 1. Tải file
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    print(f"⬇️ Đang tải file (Vui lòng đợi, file khá nặng)...")
    gdown.download(url, ZIP_FILE_NAME, quiet=False)

    # 2. Giải nén
    if os.path.exists(ZIP_FILE_NAME):
        print(f"\n📦 Đang giải nén vào thư mục: {EXTRACT_FOLDER}...")
        os.makedirs(EXTRACT_FOLDER, exist_ok=True)
        
        with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_FOLDER)

        # 3. Dọn dẹp thùng rác
        os.remove(ZIP_FILE_NAME)
        
        print("\n" + "="*60)
        print("✅ HOÀN TẤT! Mô hình đã sẵn sàng.")
        print("👉 Bây giờ hãy chạy lệnh: python Task_5_predict.py")
        print("="*60)
    else:
        print("\n❌ Tải file thất bại! Hãy kiểm tra lại link Google Drive.")

if __name__ == "__main__":
    main()