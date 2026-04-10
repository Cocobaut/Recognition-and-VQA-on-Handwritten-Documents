from Config import config
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from dataset import FolderOCRDataset
from torch.utils.data import random_split
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

Task_3_Train_Test_Config = config.return_Task3_Train_Test_Config()

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# 3. Chuyển các tham số sinh chữ sang model.generation_config
model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
model.generation_config.max_length = 64
model.generation_config.early_stopping = True
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.length_penalty = 2.0
model.generation_config.num_beams = 4

train_image_dir = Task_3_Train_Test_Config["train_image_dir"]
train_json_dir = Task_3_Train_Test_Config["train_json_dir"]
train_output_dir = Task_3_Train_Test_Config["weight"]

# Nạp dữ liệu vào biến train_dataset thông qua Class ta đã viết
full_dataset = FolderOCRDataset(
    image_dir=train_image_dir,
    json_dir=train_json_dir,
    processor=processor
)

train_size = int(0.9 * len(full_dataset))
eval_size = len(full_dataset) - train_size
# Tạo generator và set seed cố định là 42
generator = torch.Generator().manual_seed(42)
train_dataset, eval_dataset = random_split(
    full_dataset,
    [train_size, eval_size],
    generator=generator
)

print(f"Đã chia dữ liệu: {len(train_dataset)} mẫu train, {len(eval_dataset)} mẫu eval.")

# 1. Cấu hình thống số

training_args = Seq2SeqTrainingArguments(
    output_dir=train_output_dir,     # Nơi lưu đường dẫn
    save_total_limit=3,              # Tăng lên một chút để có nhiều lựa chọn checkpoint
    predict_with_generate=True,

    # Chiến lược đánh giá
    eval_strategy="steps",           # Đánh giá sau mỗi X bước
    eval_steps=500,                  # Cứ 100 bước lưu và đánh giá 1 lần
    save_steps=500,
    load_best_model_at_end=True,     # Tự động tải lại bản tốt nhất khi xong

    # Tối ưu tốc độ
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=4,  
    gradient_accumulation_steps=8,

    fp16=True, 
    bf16=False,
    learning_rate=5e-5,
    num_train_epochs=3,

    # DataLoader & Logging
    dataloader_num_workers=2,  
    logging_steps=10,
    report_to="none" 
)

# 2. Khởi tạo Trainer
trainer = Seq2SeqTrainer(
    model=model,
    processing_class=processor,
    args=training_args,
    train_dataset=train_dataset,  # 90% dữ liệu
    eval_dataset=eval_dataset,    # 10% dữ liệu dùng để đánh giá
)

# 3. Bắt đầu huấn luyện
trainer.train(resume_from_checkpoint=True)