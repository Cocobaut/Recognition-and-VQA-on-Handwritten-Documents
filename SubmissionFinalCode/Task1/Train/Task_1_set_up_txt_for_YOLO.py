import json
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import Config.config as config

# Lấy config từ file của bạn
Task_1_Config = config.return_Task1_Train_Test_Config()

def convert_json_to_obb_txt(json_dir, img_dir, output_label_dir):
    os.makedirs(output_label_dir, exist_ok=True)
    json_files = list(Path(json_dir).glob('*.json'))
    
    print(f"- Đang chuyển đổi {len(json_files)} file sang định dạng YOLO-OBB...")

    for json_file in tqdm(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']:
            temp_path = Path(img_dir) / json_file.with_suffix(ext).name
            if temp_path.exists():
                img_path = temp_path
                break
        
        if img_path is None:
            continue

        img_array = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None: continue
        h_img, w_img = img.shape[:2]

        obb_lines = []
        for block in data['output']['text_blocks']:
            p = block['polygon']
            
            """
            Chuẩn hóa tọa độ: $x_{norm} = x / W$, $y_{norm} = y / H$
            Định dạng OBB: class x0 y0 x1 y1 x2 y2 x3 y3
            """
            coords = [
                p['x0'] / w_img, p['y0'] / h_img,
                p['x1'] / w_img, p['y1'] / h_img,
                p['x2'] / w_img, p['y2'] / h_img,
                p['x3'] / w_img, p['y3'] / h_img
            ]
            
            line = "0 " + " ".join([f"{c:.6f}" for c in coords])
            obb_lines.append(line)

        txt_name = json_file.stem + ".txt"
        with open(os.path.join(output_label_dir, txt_name), 'w') as f:
            f.write("\n".join(obb_lines))

if __name__ == "__main__":
    train_labels_dir = Task_1_Config["input_images_train"].replace("images", "labels")
    convert_json_to_obb_txt(Task_1_Config["json_train"], Task_1_Config["input_images_train"], train_labels_dir)
    
    test_labels_dir = Task_1_Config["input_images_test"].replace("images", "labels")
    convert_json_to_obb_txt(Task_1_Config["json_test"], Task_1_Config["input_images_test"], test_labels_dir)