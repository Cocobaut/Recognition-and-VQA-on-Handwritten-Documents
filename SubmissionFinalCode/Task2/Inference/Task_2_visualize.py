import json
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import Config.config as config

# 1. Lấy cấu hình từ config
Task_2_Config = config.return_Task2_Predict_Config()
image_dir = Task_2_Config["input_images_test"] 
json_gt_dir = Task_2_Config["json_test"]               # Thư mục chứa Ground Truth
json_pred_dir = Task_2_Config["output_json_inference"] # Thư mục chứa kết quả của Code

# 2. Đường dẫn lưu kết quả visualize
current_script_dir = os.path.dirname(os.path.abspath(__file__))
output_visual_dir = os.path.join(current_script_dir, "Task_2_Visualization")
os.makedirs(output_visual_dir, exist_ok=True)

def imread_unicode(path):
    """Đọc ảnh từ đường dẫn chứa ký tự Unicode"""
    try:
        img_array = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except: return None

def draw_comparison():
    # Lấy danh sách file từ thư mục kết quả dự đoán
    pred_files = list(Path(json_pred_dir).glob('*.json'))
    
    print(f"- Đang vẽ so sánh GT vs Pred cho {len(pred_files)} file...")

    for p_path in tqdm(pred_files):
        # Đường dẫn file Ground Truth tương ứng (cùng tên)
        gt_path = Path(json_gt_dir) / p_path.name
        
        if not gt_path.exists():
            continue

        # Đọc dữ liệu từ cả 2 file JSON
        with open(p_path, 'r', encoding='utf-8') as f:
            data_pred = json.load(f)
        with open(gt_path, 'r', encoding='utf-8') as f:
            data_gt = json.load(f)

        # 3. Tìm ảnh gốc
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']:
            temp_path = Path(image_dir) / (p_path.stem + ext)
            if temp_path.exists():
                img_path = temp_path
                break
        
        if img_path is None: continue
        img = imread_unicode(img_path)
        if img is None: continue

        # --- A. Vẽ INPUT (Từng chữ đơn lẻ) - Màu ĐỎ (Dùng từ file GT cho chuẩn) ---
        input_words = data_gt.get('input', {}).get('text_blocks', [])
        for word in input_words:
            p = word['polygon']
            pts = np.array([[p['x0'], p['y0']], [p['x1'], p['y1']], 
                            [p['x2'], p['y2']], [p['x3'], p['y3']]], np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

        # --- B. Vẽ GROUND TRUTH (Dòng mẫu) - Màu XANH DƯƠNG ---
        # Lấy từ data_gt['output']['text_blocks']
        gt_lines = data_gt.get('output', {}).get('text_blocks', [])
        for gt in gt_lines:
            p = gt['polygon']
            pts = np.array([[p['x0'], p['y0']], [p['x1'], p['y1']], 
                            [p['x2'], p['y2']], [p['x3'], p['y3']]], np.int32).reshape((-1, 1, 2))
            
            # Vẽ nét dày màu Xanh Dương (Blue)
            cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=3)
            
            # Ghi nhãn GT ở dưới bbox
            cv2.putText(img, f"GT_{gt['id']}", (int(p['x0']), int(p['y3']) + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # --- C. Vẽ PREDICTED (Kết quả code của bạn) - Màu XANH LÁ ---
        # Lấy từ data_pred['output_predicted']['text_blocks']
        pred_lines = data_pred.get('output_predicted', {}).get('text_blocks', [])
        for pred in pred_lines:
            p = pred['polygon']
            pts = np.array([[p['x0'], p['y0']], [p['x1'], p['y1']], 
                            [p['x2'], p['y2']], [p['x3'], p['y3']]], np.int32).reshape((-1, 1, 2))
            
            # Vẽ nét màu Xanh Lá (Green)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Ghi nhãn P ở trên bbox
            cv2.putText(img, f"P_{pred['id']}", (int(p['x0']), int(p['y0']) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 4. Lưu ảnh kết quả
        save_path = os.path.join(output_visual_dir, img_path.name)
        res, img_encoded = cv2.imencode(img_path.suffix, img)
        if res:
            img_encoded.tofile(save_path)

if __name__ == "__main__":
    draw_comparison()
    print(f"\n- Hoàn thành so sánh! Xem tại: {output_visual_dir}")