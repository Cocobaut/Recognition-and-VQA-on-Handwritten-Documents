import json
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import Config.config as config

# 1. Load Cấu hình
predict_config = config.return_Task1_Predict_Config()
input_dir = predict_config["input_images"]
output_dir = predict_config["output_json"]
weight_path = os.path.join(predict_config["weight"], "weights", "best.pt")

def run_inference():
    if not os.path.exists(weight_path):
        print(f"- Không tìm thấy file trọng số tại: {weight_path}")
        return

    model = YOLO(weight_path)
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ['*.jpg', ['*.jpeg'], ['*.png'], ['*.JPG'], ['*.JPEG']]
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG']:
        image_files.extend(list(Path(input_dir).glob(ext)))

    print(f"- Bắt đầu OBB Inference cho {len(image_files)} ảnh...")

    for img_path in tqdm(image_files):
        img_array = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None: continue
        
        h_img, w_img = img.shape[:2]

        # Dự đoán OBB
        results = model.predict(source=img, imgsz=800, conf=0.25, verbose=False)
        
        text_blocks = []
        
        if results[0].obb is not None:
            obb = results[0].obb
            
            coords_pixel = obb.xyxyxyxy.cpu().numpy()
            xywhr = obb.xywhr.cpu().numpy() 

            for i in range(len(coords_pixel)):
                p = coords_pixel[i]
                s = xywhr[i]

                x_c_norm = s[0] / w_img
                y_c_norm = s[1] / h_img
                w_norm = s[2] / w_img
                h_norm = s[3] / h_img

                block = {
                    "id": i,
                    "polygon": {
                        "x0": int(p[0][0]), "y0": int(p[0][1]),
                        "x1": int(p[1][0]), "y1": int(p[1][1]),
                        "x2": int(p[2][0]), "y2": int(p[2][1]),
                        "x3": int(p[3][0]), "y3": int(p[3][1])
                    },
                    "yolo_standard": {
                        "x_center": round(float(x_c_norm), 6),
                        "y_center": round(float(y_c_norm), 6),
                        "w": round(float(w_norm), 6),
                        "h": round(float(h_norm), 6),
                        "rotation": round(float(s[4]), 4)
                    }
                }
                text_blocks.append(block)

        # Lưu file JSON
        output_data = {
            "task_info": {
                "name": "Task 1: Text Detection (OBB)",
                "total_blocks": len(text_blocks)
            },
            "output": { "text_blocks": text_blocks }
        }

        json_output_path = os.path.join(output_dir, img_path.stem + ".json")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    run_inference()