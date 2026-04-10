import os
import json
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from tqdm import tqdm

# Import file Config của dự án (Đảm bảo file Config.py nằm cùng thư mục hoặc trong PYTHONPATH)
from Config import config

# ==========================================
# 1. LẤY CẤU HÌNH ĐƯỜNG DẪN TỪ CONFIG
# ==========================================
Task_4_Predict_Config = config.return_Task4_Predict_Config()

# Trích xuất giá trị từ dictionary
TASK3_PREDICT_DIR = Task_4_Predict_Config["input_json"]      # SubmissionFinalCode/Task3/Inference/Task_3_predict_json
ORIGINAL_IMAGES_DIR = Task_4_Predict_Config["input_images"]  # dataset_project/predict_data
TASK4_WEIGHTS_DIR = Task_4_Predict_Config["weight"]          # E:/AI Competition/TextOCR/SubmissionFinalCode/Task4/Train/Weight
TASK4_OUTPUT_DIR = Task_4_Predict_Config["output_json"]      # SubmissionFinalCode/Task4/Inference/Task_4_predict_json

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(TASK4_OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. KHỞI TẠO MODEL & PROCESSOR
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Đang sử dụng thiết bị: {device}")
print("[*] Đang tải LayoutLMv3 Model và Processor từ thư mục Weight...")

processor = LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(TASK4_WEIGHTS_DIR)
model.to(device)
model.eval()

# Lấy từ điển nhãn trực tiếp từ model đã train
id2label = model.config.id2label

# ==========================================
# 3. HÀM XỬ LÝ DỮ LIỆU BỔ TRỢ
# ==========================================
def safe_normalize(polygon, w, h):
    """Chuẩn hóa tọa độ bounding box về khoảng 0-1000 cho LayoutLMv3"""
    def scale(v, max_v): return max(0, min(999, int(1000 * (v / max_v))))
    x_coords = [polygon['x0'], polygon['x1'], polygon['x2'], polygon['x3']]
    y_coords = [polygon['y0'], polygon['y1'], polygon['y2'], polygon['y3']]
    return [
        scale(min(x_coords), w), scale(min(y_coords), h),
        scale(max(x_coords), w), scale(max(y_coords), h)
    ]

# ==========================================
# 4. CHẠY DỰ ĐOÁN (INFERENCE PIPELINE)
# ==========================================
def main():
    if not os.path.exists(TASK3_PREDICT_DIR):
        print(f"[-] Lỗi: Không tìm thấy thư mục đầu vào từ Task 3: {TASK3_PREDICT_DIR}")
        return

    json_files = [f for f in os.listdir(TASK3_PREDICT_DIR) if f.endswith('.json')]
    print(f"[*] Tìm thấy {len(json_files)} file JSON từ Task 3. Bắt đầu dự đoán...")

    for json_file in tqdm(json_files, desc="Task 4 Predicting"):
        task3_json_path = os.path.join(TASK3_PREDICT_DIR, json_file)
        
        # Ánh xạ tên file JSON sang tên file ảnh (Giả định ảnh có đuôi .jpg, .jpeg, hoặc .png)
        image_id = json_file.replace('.json', '')
        
        # Thử tìm các đuôi ảnh phổ biến
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            temp_path = os.path.join(ORIGINAL_IMAGES_DIR, f"{image_id}{ext}")
            if os.path.exists(temp_path):
                image_path = temp_path
                break

        if not image_path:
            print(f"\n[-] Bỏ qua {json_file}: Không tìm thấy ảnh tương ứng tại {ORIGINAL_IMAGES_DIR}")
            continue

        # Load dữ liệu JSON từ Task 3
        with open(task3_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Mở ảnh gốc để lấy kích thước
        try:
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size
        except Exception as e:
            print(f"\n[-] Lỗi khi đọc ảnh {image_path}: {e}")
            continue

        words = []
        bboxes = []
        word_to_block_idx = [] 

        # Lấy danh sách text_blocks (Chỉnh sửa path data.get() nếu JSON cấu trúc khác)
        blocks = data.get('output', {}).get('text_blocks', [])
        if not blocks:
            blocks = data.get('input', {}).get('text_blocks', []) # <-- Phải có dòng này nó mới đọc được file JSON của bạn!
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

        # Nếu file không có chữ nào (hoặc Task 3 nhận diện rỗng) -> Lưu lại JSON nguyên bản
        if not words:
            with open(os.path.join(TASK4_OUTPUT_DIR, json_file), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            continue

        # Chuẩn bị input cho model
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

        # Chạy model dự đoán
        with torch.no_grad():
            outputs = model(**encoding)

        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        
        # Đảm bảo predictions là list (trường hợp chỉ có 1 token)
        if not isinstance(predictions, list):
            predictions = [predictions]
            
        # Gom kết quả dự đoán của từng Token về cho Block
        block_labels = {idx: [] for idx in range(len(blocks))}
        
        word_idx = 0
        for pred_idx, pred in enumerate(predictions):
            if pred != -100: # Bỏ qua token đệm padding
                if word_idx < len(word_to_block_idx):
                    block_id = word_to_block_idx[word_idx]
                    predicted_label = id2label[pred]
                    block_labels[block_id].append(predicted_label)
                    word_idx += 1

        # Cập nhật nhãn Layout vào block JSON
        for block_idx in range(len(blocks)):
            labels_in_block = block_labels.get(block_idx, [])
            if labels_in_block:
                # Majority vote: Lấy nhãn chiếm đa số trong block
                final_label = max(set(labels_in_block), key=labels_in_block.count)
            else:
                final_label = "body" # Fallback mặc định
            
            blocks[block_idx]['type'] = final_label

        # Lưu file kết quả
        output_path = os.path.join(TASK4_OUTPUT_DIR, json_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\n[+] Xong! Dữ liệu Task 4 đã được xuất thành công vào: {TASK4_OUTPUT_DIR}")

if __name__ == "__main__":
    main()