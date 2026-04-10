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
input_json = Task_5_Predict_Config["input_json"]

output_json = Task_5_Predict_Config["output_json"]

def main():
    if not torch.cuda.is_available():
        print("- Lỗi: Không tìm thấy GPU Nvidia (hoặc chưa cài đúng PyTorch CUDA)!")
        print("Chương trình đã bị ngắt để tránh việc chạy CPU quá chậm.")
        sys.exit(1) # Lệnh thoát ngang
    
    print(f"- Đã nhận diện GPU: {torch.cuda.get_device_name(0)}")

    # 2. Nạp mô hình
    print(f"- Đang tải mô hình vào Card Đồ Họa từ: {model_path}")
    
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
    json_files = glob.glob(os.path.join(input_json, "*.json"))
    print(f"- Tìm thấy {len(json_files)} file JSON đầu vào từ Task 4.\n")

    results = []

    for file_path in tqdm(json_files, desc="Tiến độ dự đoán (GPU)"):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        
        file_name = os.path.basename(file_path)
        
        if 'input' not in data or 'text_blocks' not in data['input']:
            continue
            
        full_text_blocks = [f"[{block.get('type', 'BODY').upper()}]: {block.get('text', '')}" for block in data['input']['text_blocks']]
        full_context = " \n".join(full_text_blocks)
        
        question_ids = [k for k in data['input'].keys() if k != 'text_blocks']
        file_results = {"file_name": file_name, "predictions": {}}
        
        for q_id in question_ids:
            question_text = data['input'][q_id]
            prompt = alpaca_prompt.format(question_text, full_context, "")
            
            # Quăng input vào GPU
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
        
    print(f"- Hoàn tất dự đoán")
    print(f"- File đáp án được lưu tại: {output_json}")

if __name__ == "__main__":
    main()