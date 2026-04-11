import os
import json
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from tqdm import tqdm
from Config import config

# 1. Cấu hình đường dẫn
Task_4_Predict_Config = config.return_Task4_Predict_Config()
task3_predict_dir = Task_4_Predict_Config["input_json"]
image_dir = Task_4_Predict_Config["input_images"]
weight_dir = Task_4_Predict_Config["weight"]
output_dir = Task_4_Predict_Config["output_json"]

os.makedirs(output_dir, exist_ok=True)

# 2. Khởi tạo mô hình và processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Đang sử dụng thiết bị: {device}")

# Ưu tiên load processor từ thư mục weight để đảm bảo tính đồng bộ offline
try:
    processor = LayoutLMv3Processor.from_pretrained(weight_dir, apply_ocr=False)
except:
    processor = LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False)

model = LayoutLMv3ForTokenClassification.from_pretrained(weight_dir)
model.to(device)
model.eval()

id2label = model.config.id2label

# 3. Hàm xử lí và phụ trợ
def safe_normalize(polygon, w, h):
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
        print(f"[-] Lỗi: Không tìm thấy thư mục đầu vào: {task3_predict_dir}")
        return

    json_files = [f for f in os.listdir(task3_predict_dir) if f.endswith('.json')]
    
    for json_file in tqdm(json_files, desc="Task 4 Predicting"):
        task3_json_path = os.path.join(task3_predict_dir, json_file)
        image_id = json_file.replace('.json', '')
        
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            temp_path = os.path.join(image_dir, f"{image_id}{ext}")
            if os.path.exists(temp_path):
                image_path = temp_path
                break

        if not image_path: continue

        with open(task3_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size

        # CHỈNH SỬA 1: Tìm đúng key chứa text từ Task 3
        if "output_predicted" in data and "text_blocks" in data["output_predicted"]:
            blocks = data["output_predicted"]["text_blocks"]
        elif "input" in data and "text_blocks" in data["input"]:
            blocks = data["input"]["text_blocks"]
        else:
            blocks = data.get('text_blocks', [])

        if not blocks:
            # Lưu lại file gốc nếu không có blocks để xử lý
            with open(os.path.join(output_dir, json_file), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            continue

        words = []
        bboxes = []
        word_to_block_idx = [] 

        for block_idx, block in enumerate(blocks):
            text = block.get('text', '')
            word_list = text.split()
            if not word_list: 
                # Trường hợp text rỗng, gán mặc định để tránh lỗi logic
                word_list = ["%NA%"]
            
            polygon = block.get('polygon', {})
            bbox = safe_normalize(polygon, img_width, img_height)
            
            for w in word_list:
                words.append(w)
                bboxes.append(bbox)
                word_to_block_idx.append(block_idx)

        # Encode dữ liệu
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

        # CHỈNH SỬA 2: Sử dụng word_ids để map nhãn chuẩn xác (Tránh lệch do tách từ)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        word_ids = encoding.word_ids(batch_index=0) 
        
        block_labels = {idx: [] for idx in range(len(blocks))}
        
        for pred_idx, word_idx in enumerate(word_ids):
            # word_idx tương ứng với index trong danh sách 'words' ban đầu
            if word_idx is not None:
                label_id = predictions[pred_idx]
                # Bỏ qua nhãn -100 (padding/special tokens)
                if label_id != -100:
                    label_name = id2label[label_id]
                    actual_block_id = word_to_block_idx[word_idx]
                    block_labels[actual_block_id].append(label_name)

        # CHỈNH SỬA 3: Gán nhãn cho từng block bằng Majority Voting
        for block_idx, block in enumerate(blocks):
            labels_in_block = block_labels.get(block_idx, [])
            if labels_in_block:
                final_label = max(set(labels_in_block), key=labels_in_block.count)
            else:
                final_label = "body" # Default label
            
            block['type'] = final_label

        # Lưu kết quả
        output_path = os.path.join(output_dir, json_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\n[+] Xong! Dữ liệu Task 4 đã được xuất tại: {output_dir}")

if __name__ == "__main__":
    main()