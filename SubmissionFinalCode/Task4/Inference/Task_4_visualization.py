import os
import json
from PIL import Image, ImageDraw, ImageFont
from Config import config

# Cấu hình đường dẫn
Task_4_Predict_Config = config.return_Task4_Predict_Config()
json_dir = Task_4_Predict_Config["output_json"]                             # Nơi chứa JSON kết quả
image_dir = Task_4_Predict_Config["input_images"]                           # Nơi chứa ảnh gốc
output_dir = "SubmissionFinalCode/Task4/Inference/Task_4_Visualization"

os.makedirs(output_dir, exist_ok=True)

json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

for json_file in json_files:
    image_id = json_file.replace('.json', '')
    
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        temp_path = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(temp_path):
            img_path = temp_path
            break
            
    if not img_path:
        continue
        
    with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
    except Exception as e:
        print(f"Error open picture: {e}")
        continue
        
    blocks = data.get('output', {}).get('text_blocks', [])
    if not blocks: blocks = data.get('input', {}).get('text_blocks', [])
    if not blocks: blocks = data.get('text_blocks', [])
    
    for block in blocks:
        poly = block.get('polygon', {})
        label = block.get('type', 'Unknown')
        
        if poly:
            coords = [
                (poly.get('x0', 0), poly.get('y0', 0)),
                (poly.get('x1', 0), poly.get('y1', 0)),
                (poly.get('x2', 0), poly.get('y2', 0)),
                (poly.get('x3', 0), poly.get('y3', 0))
            ]
            
            color = "red"
            if label.lower() == 'title': color = "orange"
            elif label.lower() in ['body', 'p']: color = "blue"
            elif label.lower() == 'list': color = "green"
            
            draw.polygon(coords, outline=color, width=4)
            
            x0, y0 = coords[0]
            draw.rectangle([x0, max(0, y0-25), x0+100, y0], fill="black")
            draw.text((x0+5, max(0, y0-20)), label.upper(), fill="yellow")

    out_path = os.path.join(output_dir, f"{image_id}_visualized.jpg")
    img.save(out_path)
    print(f"Visualize and Save in: {out_path}")

print("-Success")