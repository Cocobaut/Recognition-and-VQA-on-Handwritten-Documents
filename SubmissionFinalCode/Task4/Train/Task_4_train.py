import os
import json
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    LayoutLMv3Processor, 
    LayoutLMv3ForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)

# Import file Config của dự án
from Config import config

# ==========================================
# 1. LẤY CẤU HÌNH TỪ CONFIG
# ==========================================
Task_4_Train_Config = config.return_Task4_Train_Test_Config()

TRAIN_JSON_DIR = Task_4_Train_Config["json_train"]  # dataset_project/train_data/task_4
TEST_JSON_DIR = Task_4_Train_Config["json_test"]    # dataset_project/test_data/task_4
WEIGHT_OUTPUT_DIR = Task_4_Train_Config["weight"]   # E:/AI Competition/TextOCR/SubmissionFinalCode/Task4/Train/Weight

# Tạo thư mục lưu weight nếu chưa có
os.makedirs(WEIGHT_OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. HÀM XỬ LÝ DỮ LIỆU
# ==========================================
def normalize_bbox(polygon, max_x=4230, max_y=4230):
    """Chuẩn hóa tọa độ bounding box về khoảng 0-1000 cho LayoutLMv3"""
    x_coords = [polygon['x0'], polygon['x1'], polygon['x2'], polygon['x3']]
    y_coords = [polygon['y0'], polygon['y1'], polygon['y2'], polygon['y3']]
    def clip(v): return max(0, min(999, int(v)))
    
    x0 = clip(1000 * (min(x_coords) / max_x))
    y0 = clip(1000 * (min(y_coords) / max_y))
    x1 = clip(1000 * (max(x_coords) / max_x))
    y1 = clip(1000 * (max(y_coords) / max_y))
    return [x0, y0, x1, y1]

def load_data(data_dir, limit=None, is_balanced=False):
    """Đọc dữ liệu từ thư mục JSON và trích xuất words, bboxes, labels"""
    data = []
    label_set = set()
    
    if not os.path.exists(data_dir):
        print(f"[-] Không tìm thấy thư mục: {data_dir}")
        return data, []

    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    print(f"[*] Đang đọc dữ liệu từ {data_dir} ({len(files)} files)...")

    for file in tqdm(files, desc=f'Loading {os.path.basename(data_dir)}'):
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
            item = json.load(f)
            
        words, bboxes, labels = [], [], []
        
        # Lấy text_blocks
        blocks = item.get('output', {}).get('text_blocks', [])
        if not blocks:
            blocks = item.get('input', {}).get('text_blocks', [])
            
        if not blocks:
            blocks = item.get('text_blocks', [])

        has_diverse_label = False
        temp_words, temp_bboxes, temp_labels = [], [], []

        for block in blocks:
            text = block.get('text', '')
            word_list = text.split()
            if not word_list: continue
            
            label = block.get('type', 'body')
            if label.lower() not in ['body', 'p']: 
                has_diverse_label = True

            polygon = block.get('polygon')
            if not polygon: continue
            
            bbox = normalize_bbox(polygon)
            
            temp_words.extend(word_list)
            temp_bboxes.extend([bbox] * len(word_list))
            temp_labels.extend([label] * len(word_list))
            label_set.add(label)

        # Logic cân bằng dữ liệu (giữ lại từ code Colab của bạn)
        if is_balanced:
            if has_diverse_label or (limit and len(data) < limit / 2):
                data.append({'words': temp_words, 'bboxes': temp_bboxes, 'labels': temp_labels})
        else:
            if temp_words:
                data.append({'words': temp_words, 'bboxes': temp_bboxes, 'labels': temp_labels})

        if limit and len(data) >= limit: 
            break

    return data, sorted(list(label_set))

# ==========================================
# 3. CHẠY PIPELINE HUẤN LUYỆN
# ==========================================
def main():
    # Load data
    print("--- CHUẨN BỊ DỮ LIỆU ---")
    train_samples, train_labels = load_data(TRAIN_JSON_DIR, limit=2000, is_balanced=True)
    test_samples, test_labels = load_data(TEST_JSON_DIR, limit=None, is_balanced=False)

    if not train_samples:
        print("[-] Dữ liệu train rỗng. Vui lòng kiểm tra lại đường dẫn.")
        return

    # Gom tất cả các nhãn lại để tạo dictionary
    all_labels = sorted(list(set(train_labels + test_labels)))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for i, l in enumerate(all_labels)}
    
    print(f"[*] Phát hiện {len(all_labels)} nhãn: {all_labels}")

    # Khởi tạo Processor
    print("--- KHỞI TẠO MODEL & PROCESSOR ---")
    processor = LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False)

    def preprocess_data(examples):
        return processor.tokenizer(
            examples['words'], 
            boxes=examples['bboxes'], 
            word_labels=[[label2id[l] for l in ls] for ls in examples['labels']], 
            truncation=True, 
            padding='max_length', 
            max_length=512
        )

    # Đóng gói vào HuggingFace Dataset
    train_dataset = Dataset.from_list(train_samples)
    encoded_train = train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names)
    encoded_train.set_format('torch')

    if test_samples:
        test_dataset = Dataset.from_list(test_samples)
        encoded_test = test_dataset.map(preprocess_data, batched=True, remove_columns=test_dataset.column_names)
        encoded_test.set_format('torch')
    else:
        # Nếu không có tập test thực tế, trích xuất 10% từ tập train
        print("[!] Không tìm thấy tập test. Tự động chia 10% từ tập train để đánh giá.")
        dataset = train_dataset.train_test_split(test_size=0.1)
        encoded_train = dataset['train'].map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
        encoded_test = dataset['test'].map(preprocess_data, batched=True, remove_columns=dataset['test'].column_names)
        encoded_train.set_format('torch')
        encoded_test.set_format('torch')

    # Khởi tạo Model
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        'microsoft/layoutlmv3-base', 
        num_labels=len(all_labels), 
        id2label=id2label, 
        label2id=label2id
    )

    # Cấu hình tham số Training
    training_args = TrainingArguments(
        output_dir='./tmp_layoutlmv3_checkpoints', # Thư mục tạm chứa checkpoints
        max_steps=800,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        evaluation_strategy='steps',
        eval_steps=200,
        logging_steps=50,
        report_to='none',
        save_strategy='steps',
        save_steps=200,
        load_best_model_at_end=True # Tự động load model tốt nhất ở cuối
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=encoded_train, 
        eval_dataset=encoded_test, 
        data_collator=DataCollatorForTokenClassification(processor.tokenizer)
    )

    # Bắt đầu Train
    print("--- BẮT ĐẦU HUẤN LUYỆN ---")
    trainer.train()

    # Đánh giá sau khi train xong
    print("--- ĐÁNH GIÁ MÔ HÌNH ---")
    metrics = trainer.evaluate()
    print(metrics)

    # Lưu model và processor vào thư mục config đã chỉ định
    print(f"--- LƯU TRỌNG SỐ (WEIGHTS) VÀO: {WEIGHT_OUTPUT_DIR} ---")
    trainer.save_model(WEIGHT_OUTPUT_DIR)
    processor.save_pretrained(WEIGHT_OUTPUT_DIR)
    print("[+] Hoàn thành! Quá trình Train Task 4 đã xong.")

if __name__ == "__main__":
    main()