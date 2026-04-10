import os
import json
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
JSON_DIR = "SubmissionFinalCode/Task4/Inference/Task_4_predict_json" # Nơi chứa JSON kết quả
IMAGE_DIR = "dataset_project/predict_data"                           # Nơi chứa ảnh gốc
OUTPUT_VIS_DIR = "SubmissionFinalCode/Task4/Inference/Visualized"    # Thư mục lưu ảnh đã vẽ

os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

print("--- BẮT ĐẦU VẼ ẢNH ---")
json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]

for json_file in json_files:
    image_id = json_file.replace('.json', '')
    
    # 1. Tìm ảnh gốc tương ứng
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        temp_path = os.path.join(IMAGE_DIR, f"{image_id}{ext}")
        if os.path.exists(temp_path):
            img_path = temp_path
            break
            
    if not img_path:
        continue
        
    # 2. Đọc JSON kết quả của Task 4
    with open(os.path.join(JSON_DIR, json_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 3. Mở ảnh để vẽ
    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
    except Exception as e:
        print(f"Lỗi mở ảnh: {e}")
        continue
        
    blocks = data.get('output', {}).get('text_blocks', [])
    if not blocks: blocks = data.get('input', {}).get('text_blocks', [])
    if not blocks: blocks = data.get('text_blocks', [])
    
    # 4. Vẽ từng block
    for block in blocks:
        poly = block.get('polygon', {})
        label = block.get('type', 'Unknown') # Nhãn LayoutLMv3 đã dự đoán
        
        if poly:
            # Lấy tọa độ 4 góc
            coords = [
                (poly.get('x0', 0), poly.get('y0', 0)),
                (poly.get('x1', 0), poly.get('y1', 0)),
                (poly.get('x2', 0), poly.get('y2', 0)),
                (poly.get('x3', 0), poly.get('y3', 0))
            ]
            
            # Tô màu tùy theo nhãn cho dễ nhìn (thêm bớt tùy ý)
            color = "red"
            if label.lower() == 'title': color = "orange"
            elif label.lower() in ['body', 'p']: color = "blue"
            elif label.lower() == 'list': color = "green"
            
            # Vẽ viền (polygon) xung quanh chữ
            draw.polygon(coords, outline=color, width=4)
            
            # Viết cái chữ nhãn (Label) lên trên góc trái của hộp
            # Chỉnh nền đen chữ vàng cho dễ đọc
            x0, y0 = coords[0]
            draw.rectangle([x0, max(0, y0-25), x0+100, y0], fill="black")
            draw.text((x0+5, max(0, y0-20)), label.upper(), fill="yellow")

    # 5. Lưu ảnh
    out_path = os.path.join(OUTPUT_VIS_DIR, f"{image_id}_visualized.jpg")
    img.save(out_path)
    print(f"[+] Đã vẽ và lưu ảnh: {out_path}")

print("--- HOÀN THÀNH ---")