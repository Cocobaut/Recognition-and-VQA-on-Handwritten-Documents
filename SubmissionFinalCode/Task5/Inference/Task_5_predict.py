import os
import sys
import json
import glob
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model")
INPUT_JSON_DIR = os.path.join(BASE_DIR, "..", "..", "Task4", "predict")

OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "Task5", "predict")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Task_5_predict.json")

def main():
    # ==============================================================================
    # KIỂM TRA GPU - NẾU KHÔNG CÓ THÌ NGẮT NGAY LẬP TỨC!
    # ==============================================================================
    if not torch.cuda.is_available():
        print("❌ LỖI CHÍ MẠNG: Không tìm thấy GPU Nvidia (hoặc chưa cài đúng PyTorch CUDA)!")
        print("Chương trình đã bị ngắt để tránh việc chạy CPU quá chậm.")
        sys.exit(1) # Lệnh thoát ngang
    
    print(f"🔥 Đã nhận diện GPU: {torch.cuda.get_device_name(0)}")

    # ==============================================================================
    # 2. NẠP MÔ HÌNH VÀO GPU BẰNG PEFT + 4-BIT
    # ==============================================================================
    print(f"🔄 Đang tải mô hình vào Card Đồ Họa từ: {MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Ép 4-bit để chạy mượt trên RTX 4060 (8GB VRAM)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map="cuda" # Ép chặt vào GPU
    )
    model.eval()

    alpaca_prompt = """Dưới đây là một yêu cầu. Hãy sử dụng thông tin trong phần Ngữ cảnh để trả lời thật chính xác.\n\n### Yêu cầu:\n{}\n\n### Ngữ cảnh (Layout JSON):\n{}\n\n### Trả lời:\n{}"""

    # ==============================================================================
    # 3. QUÉT FILE VÀ DỰ ĐOÁN
    # ==============================================================================
    json_files = glob.glob(os.path.join(INPUT_JSON_DIR, "*.json"))
    print(f"🔍 Tìm thấy {len(json_files)} file JSON đầu vào từ Task 4.\n")

    results = []

    for file_path in tqdm(json_files, desc="Tiến độ dự đoán (GPU 🚀)"):
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
        
    # ==============================================================================
    # 4. LƯU KẾT QUẢ
    # ==============================================================================
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print("\n" + "="*60)
    print(f"✅ HOÀN TẤT DỰ ĐOÁN SIÊU TỐC!")
    print(f"💾 File đáp án được lưu tại: {OUTPUT_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()