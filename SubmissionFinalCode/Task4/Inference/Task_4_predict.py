import os
import json
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from tqdm import tqdm
from Config import config

# 1. Cấu hình đường dẫn
Task_4_Predict_Config = config.return_Task4_Predict_Config()

# Trích xuất giá trị từ dictionary
task3_predict_dir = Task_4_Predict_Config["input_json"]
image_dir = Task_4_Predict_Config["input_images"]
weight_dir = Task_4_Predict_Config["weight"]
output_dir = Task_4_Predict_Config["output_json"]

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)

#2. Khởi tạo mô hình và processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Đang sử dụng thiết bị: {device}")
print("[*] Đang tải LayoutLMv3 Model và Processor từ thư mục Weight...")

processor = LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(weight_dir)
model.to(device)
model.eval()

# Lấy từ điển nhãn trực tiếp từ model đã train
id2label = model.config.id2label

# 3. Hàm xử lí và phụ trợ
def safe_normalize(polygon, w, h):
    """Chuẩn hóa tọa độ bounding box về khoảng 0-1000 cho LayoutLMv3"""
    def scale(v, max_v): return max(0, min(999, int(1000 * (v / max_v))))
    x_coords = [polygon['x0'], polygon['x1'], polygon['x2'], polygon['x3']]
    y_coords = [polygon['y0'], polygon['y1'], polygon['y2'], polygon['y3']]
    return [
        scale(min(x_coords), w), scale(min(y_coords), h),
        scale(max(x_coords), w), scale(max(y_coords), h)
    ]

# 4. Chạy dự đoán
def main():
    if not os.path.exists(task3_predict_dir):
        print(f"[-] Lỗi: Không tìm thấy thư mục đầu vào từ Task 3: {task3_predict_dir}")
        return

    json_files = [f for f in os.listdir(task3_predict_dir) if f.endswith('.json')]
    print(f"[*] Tìm thấy {len(json_files)} file JSON từ Task 3. Bắt đầu dự đoán...")

    for json_file in tqdm(json_files, desc="Task 4 Predicting"):
        task3_json_path = os.path.join(task3_predict_dir, json_file)
        
        image_id = json_file.replace('.json', '')
        
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            temp_path = os.path.join(image_dir, f"{image_id}{ext}")
            if os.path.exists(temp_path):
                image_path = temp_path
                break

        if not image_path:
            print(f"\n[-] Bỏ qua {json_file}: Không tìm thấy ảnh tương ứng tại {image_dir}")
            continue

        with open(task3_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        try:
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size
        except Exception as e:
            print(f"\n[-] Lỗi khi đọc ảnh {image_path}: {e}")
            continue

        words = []
        bboxes = []
        word_to_block_idx = [] 

        blocks = data.get('output', {}).get('text_blocks', [])
        if not blocks:
            blocks = data.get('input', {}).get('text_blocks', [])
        if not blocks:
            blocks = data.get('text_blocks', [])

        for block_idx, block in enumerate(blocks):
            text = block.get('text', '')
            word_list = text.split()
            if not word_list: continue
            
            polygon = block.get('polygon', {})
            if not polygon: continue 
            
            bbox = safe_normalize(polygon, img_width, img_height)
            
            words.extend(word_list)
            bboxes.extend([bbox] * len(word_list))
            word_to_block_idx.extend([block_idx] * len(word_list))

        if not words:
            with open(os.path.join(output_dir, json_file), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            continue

        encoding = processor(
            image, 
            text=words, 
            boxes=bboxes, 
            return_tensors="pt", 
            truncation=True, 
            padding="max_length"
        )
        
        for k, v in encoding.items():
            encoding[k] = v.to(device)

        with torch.no_grad():
            outputs = model(**encoding)

        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        
        if not isinstance(predictions, list):
            predictions = [predictions]
            
        block_labels = {idx: [] for idx in range(len(blocks))}
        
        word_idx = 0
        for pred_idx, pred in enumerate(predictions):
            if pred != -100:
                if word_idx < len(word_to_block_idx):
                    block_id = word_to_block_idx[word_idx]
                    predicted_label = id2label[pred]
                    block_labels[block_id].append(predicted_label)
                    word_idx += 1

        for block_idx in range(len(blocks)):
            labels_in_block = block_labels.get(block_idx, [])
            if labels_in_block:
                final_label = max(set(labels_in_block), key=labels_in_block.count)
            else:
                final_label = "body"
            
            blocks[block_idx]['type'] = final_label

        output_path = os.path.join(output_dir, json_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\n[+] Xong! Dữ liệu Task 4 đã được xuất thành công vào: {output_dir}")

if __name__ == "__main__":
    main()