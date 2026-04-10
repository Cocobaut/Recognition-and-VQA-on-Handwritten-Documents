import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    LayoutLMv3Processor, 
    LayoutLMv3ForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import file Config của dự án
from Config import config

# ==========================================
# 1. LẤY CẤU HÌNH TỪ CONFIG
# ==========================================
Task_4_Train_Config = config.return_Task4_Train_Test_Config()

TRAIN_JSON_DIR = Task_4_Train_Config["json_train"]      # dataset_project/train_data/task_4
TEST_JSON_DIR = Task_4_Train_Config["json_test"]        # dataset_project/test_data/task_4
TRAIN_IMG_DIR = Task_4_Train_Config["input_images_train"] # dataset_project/train_data/images
TEST_IMG_DIR = Task_4_Train_Config["input_images_test"]   # dataset_project/test_data/images
WEIGHT_OUTPUT_DIR = Task_4_Train_Config["weight"]       # SubmissionFinalCode/Task4/Train/Weight

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

def load_data(data_dir, img_dir, limit=None, is_balanced=False):
    """Đọc dữ liệu từ JSON và TÌM ẢNH GỐC tương ứng"""
    data = []
    label_set = set()
    
    if not os.path.exists(data_dir):
        print(f"[-] Không tìm thấy thư mục JSON: {data_dir}")
        return data, []

    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    print(f"[*] Đang đọc dữ liệu từ {data_dir} ({len(files)} files)...")

    for file in tqdm(files, desc=f'Loading {os.path.basename(data_dir)}'):
        # --- TÌM ẢNH TƯƠNG ỨNG ---
        image_id = file.replace('.json', '')
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            temp_path = os.path.join(img_dir, f"{image_id}{ext}")
            if os.path.exists(temp_path):
                img_path = temp_path
                break
                
        if not img_path:
            continue # Bỏ qua nếu JSON không có ảnh đi kèm
            
        # --- ĐỌC JSON ---
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
            item = json.load(f)
            
        words, bboxes, labels = [], [], []
        
        blocks = item.get('output', {}).get('text_blocks', [])
        if not blocks: blocks = item.get('input', {}).get('text_blocks', [])
        if not blocks: blocks = item.get('text_blocks', [])

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

        if is_balanced:
            if has_diverse_label or (limit and len(data) < limit / 2):
                data.append({'words': temp_words, 'bboxes': temp_bboxes, 'labels': temp_labels, 'image_path': img_path})
        else:
            if temp_words:
                data.append({'words': temp_words, 'bboxes': temp_bboxes, 'labels': temp_labels, 'image_path': img_path})

        if limit and len(data) >= limit: 
            break

    return data, sorted(list(label_set))

# ==========================================
# 3. CHẠY PIPELINE HUẤN LUYỆN
# ==========================================
def main():
    print("--- CHUẨN BỊ DỮ LIỆU ---")
    train_samples, train_labels = load_data(TRAIN_JSON_DIR, TRAIN_IMG_DIR, limit=2000, is_balanced=True)
    test_samples, test_labels = load_data(TEST_JSON_DIR, TEST_IMG_DIR, limit=None, is_balanced=False)

    if not train_samples:
        print("[-] Dữ liệu train rỗng. Vui lòng kiểm tra lại đường dẫn ảnh và JSON.")
        return

    all_labels = sorted(list(set(train_labels + test_labels)))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for i, l in enumerate(all_labels)}
    
    print(f"[*] Phát hiện {len(all_labels)} nhãn: {all_labels}")

    print("--- KHỞI TẠO MODEL & PROCESSOR ---")
    processor = LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False)

    # ĐÃ SỬA: Load thêm ảnh truyền vào processor
    def preprocess_data(examples):
        images = [Image.open(path).convert("RGB") for path in examples['image_path']]
        return processor(
            images, # <-- LayoutLMv3 đã được "mở mắt" nhìn ảnh
            text=examples['words'], 
            boxes=examples['bboxes'], 
            word_labels=[[label2id[l] for l in ls] for ls in examples['labels']], 
            truncation=True, 
            padding='max_length', 
            max_length=512
        )

    train_dataset = Dataset.from_list(train_samples)
    encoded_train = train_dataset.map(preprocess_data, batched=True, batch_size=8, remove_columns=train_dataset.column_names)
    encoded_train.set_format('torch')

    if test_samples:
        test_dataset = Dataset.from_list(test_samples)
        encoded_test = test_dataset.map(preprocess_data, batched=True, batch_size=8, remove_columns=test_dataset.column_names)
        encoded_test.set_format('torch')
    else:
        print("[!] Không tìm thấy tập test. Tự động chia 10% từ tập train để đánh giá.")
        dataset = train_dataset.train_test_split(test_size=0.1)
        encoded_train = dataset['train']
        encoded_test = dataset['test']

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        'microsoft/layoutlmv3-base', 
        num_labels=len(all_labels), 
        id2label=id2label, 
        label2id=label2id
    )

    model.to('cuda')
    
    # --- ĐÃ THÊM: HÀM TÍNH ĐIỂM F1, ACCURACY ---
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Loại bỏ các token padding (-100) để không làm sai lệch điểm số
        true_predictions = [
            id2label[p] for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100
        ]
        true_labels = [
            id2label[l] for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100
        ]

        # Tính điểm bằng sklearn (trung bình weighted)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average='weighted', zero_division=0)
        acc = accuracy_score(true_labels, true_predictions)

        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    training_args = TrainingArguments(
        output_dir='./tmp_layoutlmv3_checkpoints',
        max_steps=800,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        evaluation_strategy='steps',
        eval_steps=200,
        logging_steps=50,
        report_to='none',
        save_strategy='steps',
        save_steps=200,
        load_best_model_at_end=True 
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=encoded_train, 
        eval_dataset=encoded_test, 
        data_collator=DataCollatorForTokenClassification(processor.tokenizer),
        compute_metrics=compute_metrics # <-- Tích hợp bảng điểm vào Trainer
    )

    print("--- BẮT ĐẦU HUẤN LUYỆN ---")
    trainer.train()

    print("--- ĐÁNH GIÁ MÔ HÌNH ---")
    metrics = trainer.evaluate()
    
    # In bảng điểm ra cho đẹp
    print("\n" + "="*40)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
    print("="*40)
    print(f"Accuracy  : {metrics.get('eval_accuracy', 0):.4f}")
    print(f"F1-Score  : {metrics.get('eval_f1', 0):.4f}")
    print(f"Precision : {metrics.get('eval_precision', 0):.4f}")
    print(f"Recall    : {metrics.get('eval_recall', 0):.4f}")
    print(f"Loss      : {metrics.get('eval_loss', 0):.4f}")
    print("="*40 + "\n")

    print(f"--- LƯU TRỌNG SỐ (WEIGHTS) VÀO: {WEIGHT_OUTPUT_DIR} ---")
    trainer.save_model(WEIGHT_OUTPUT_DIR)
    processor.save_pretrained(WEIGHT_OUTPUT_DIR)
    print("[+] Hoàn thành! Quá trình Train Task 4 đã xong.")

if __name__ == "__main__":
    main()