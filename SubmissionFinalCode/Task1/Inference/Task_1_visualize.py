import json
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import Config.config as config

# 1. Lấy thông tin từ config
predict_config = config.return_Task1_Predict_Config()
image_dir = predict_config["input_images"] 
json_dir = predict_config["output_json"]   

output_visual_dir = "SubmissionFinalCode/Task1/Inference/Task_1_OBB_Visualization"

def draw_obb_results():
    os.makedirs(output_visual_dir, exist_ok=True)
    
    # Lấy danh sách file JSON đã dự đoán
    json_files = list(Path(json_dir).glob('*.json'))
    if not json_files:
        print(f"- Không tìm thấy file JSON nào tại: {json_dir}")
        return

    print(f"- Đang vẽ OBB Visualization cho {len(json_files)} file...")

    for json_path in tqdm(json_files):
        # 2. Đọc dữ liệu JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 3. Tìm ảnh tương ứng (Hỗ trợ nhiều định dạng)
        img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']
        img_path = None
        for ext in img_extensions:
            temp_path = Path(image_dir) / (json_path.stem + ext)
            if temp_path.exists():
                img_path = temp_path
                break
        
        if img_path is None:
            continue

        img_array = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None: continue

        # 4. Vẽ từng block chữ (OBB)
        blocks = data.get('output', {}).get('text_blocks', [])
        
        for block in blocks:
            p = block.get('polygon', {})
            bid = block.get('id', '?')
            
            pts = np.array([
                [p['x0'], p['y0']], 
                [p['x1'], p['y1']], 
                [p['x2'], p['y2']], 
                [p['x3'], p['y3']]
            ], np.int32)
            
            pts = pts.reshape((-1, 1, 2))

            # Vẽ đường bao OBB:
            # - Màu xanh lá (0, 255, 0)
            # - Độ dày 2
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Vẽ điểm gốc (x0, y0) màu đỏ để biết hướng của Bbox
            cv2.circle(img, (int(p['x0']), int(p['y0'])), 3, (0, 0, 255), -1)

            cv2.putText(img, str(bid), (int(p['x0']), int(p['y0']) - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

        # 5. Lưu ảnh kết quả (Unicode an toàn)
        output_img_path = os.path.join(output_visual_dir, img_path.name)
        ext = img_path.suffix
        res, img_encoded = cv2.imencode(ext, img)
        if res:
            img_encoded.tofile(output_img_path)

if __name__ == '__main__':
    draw_obb_results()
    print(f"\n- Đã vẽ xong! Bạn hãy kiểm tra các hộp xoay tại: {output_visual_dir}")