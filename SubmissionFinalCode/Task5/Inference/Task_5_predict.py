import os
import sys
import json
import glob
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from Config import config

Task_5_Predict_Config = config.return_Task5_Predict_Config()

# 1. Cấu hình đường dẫn
model_path = os.path.join(Task_5_Predict_Config["weight"], "model", "model")
input_json = Task_5_Predict_Config["input_json"]   # Nơi chứa context (layout)
input_text = Task_5_Predict_Config["input_text"]   # Nơi chứa câu hỏi (questions)
output_json = Task_5_Predict_Config["output_json"]

def main():
    if not torch.cuda.is_available():
        print("- Lỗi: Không tìm thấy GPU Nvidia!")
        sys.exit(1)
    
    print(f"- Đã nhận diện GPU: {torch.cuda.get_device_name(0)}")

    # 2. Nạp mô hình
    print(f"- Đang tải mô hình từ: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="cuda"
    )
    model.eval()

    alpaca_prompt = """Dưới đây là một yêu cầu. Hãy sử dụng thông tin trong phần Ngữ cảnh để trả lời thật chính xác.\n\n### Yêu cầu:\n{}\n\n### Ngữ cảnh (Layout JSON):\n{}\n\n### Trả lời:\n{}"""

    # 3. Dự đoán
    # Lấy danh sách file từ input_json (thư mục chứa layout)
    json_files = glob.glob(os.path.join(input_json, "*.json"))
    print(f"- Tìm thấy {len(json_files)} file dữ liệu đầu vào.\n")

    results = []

    for context_file_path in tqdm(json_files, desc="Tiến độ dự đoán (GPU)"):
        file_name = os.path.basename(context_file_path)
        question_file_path = os.path.join(input_text, file_name)

        # Kiểm tra xem file câu hỏi tương ứng có tồn tại không
        if not os.path.exists(question_file_path):
            print(f"\n- Cảnh báo: Không tìm thấy file câu hỏi cho {file_name}, bỏ qua.")
            continue

        # Đọc dữ liệu ngữ cảnh (Context)
        with open(context_file_path, 'r', encoding='utf-8') as f:
            try:
                context_data = json.load(f)
            except json.JSONDecodeError:
                continue

        # Đọc dữ liệu câu hỏi (Questions)
        with open(question_file_path, 'r', encoding='utf-8') as f:
            try:
                questions_dict = json.load(f)
            except json.JSONDecodeError:
                continue
        
        # Trích xuất text blocks để làm context
        if 'input' not in context_data or 'text_blocks' not in context_data['input']:
            continue
            
        full_text_blocks = [f"[{block.get('type', 'BODY').upper()}]: {block.get('text', '')}" for block in context_data['input']['text_blocks']]
        full_context = " \n".join(full_text_blocks)
        
        file_results = {"file_name": file_name, "predictions": {}}
        
        # Duyệt qua từng câu hỏi trong file input_text
        for q_id, question_text in questions_dict.items():
            prompt = alpaca_prompt.format(question_text, full_context, "")
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            input_length = inputs.input_ids.shape[1]
            cau_tra_loi_ai = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            
            file_results["predictions"][q_id] = {
                "question": question_text,
                "answer": cau_tra_loi_ai
            }
            
        results.append(file_results)
        
    # 4. Lưu kết quả
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"- Hoàn tất dự đoán. File lưu tại: {output_json}")

if __name__ == "__main__":
    main()