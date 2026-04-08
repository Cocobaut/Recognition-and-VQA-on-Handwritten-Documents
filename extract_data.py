import json
import os
from pathlib import Path

def process_to_task5_refined():
    # --- 1. Cấu hình đường dẫn ---
    path_task4 = Path(r"E:\AI Competition\TextOCR\dataset_project\test_data\task_4")
    path_squad = Path(r"E:\AI Competition\TextOCR\dataset_set_up\dataset_HW-SQuAD\HW-SQuAD_annotations_normalization\test")
    path_task5 = Path(r"E:\AI Competition\TextOCR\dataset_project\test_data\task_5")

    path_task5.mkdir(parents=True, exist_ok=True)

    task4_files = list(path_task4.glob("*.json"))
    print(f"🚀 Đang xử lý {len(task4_files)} files...")

    for f4_path in task4_files:
        file_name = f4_path.name
        f_squad_path = path_squad / file_name
        
        # --- 2. Lấy DUY NHẤT nội dung text_blocks từ Output của Task 4 ---
        clean_text_blocks = []
        try:
            with open(f4_path, 'r', encoding='utf-8') as f:
                data_t4 = json.load(f)
                
                # Truy cập vào: output -> text_blocks
                # Đây là nơi chứa id, polygon, text, type mà bạn cần
                if "output" in data_t4 and "text_blocks" in data_t4["output"]:
                    clean_text_blocks = data_t4["output"]["text_blocks"]
                else:
                    # Trường hợp file task 4 có cấu trúc khác, thử lấy trực tiếp
                    clean_text_blocks = data_t4.get("text_blocks", [])
        except Exception as e:
            print(f"❌ Lỗi đọc Task 4 ({file_name}): {e}")
            continue

        # --- 3. Khởi tạo cấu trúc Task 5 sạch ---
        result = {
            "task_info": {
                "name": "Task 5: LLM",
                "total_blocks": len(clean_text_blocks)
            },
            "input": {
                "text_blocks": clean_text_blocks
            },
            "output": {}
        }

        # --- 4. Khớp dữ liệu câu hỏi và câu trả lời ---
        if f_squad_path.exists():
            try:
                with open(f_squad_path, 'r', encoding='utf-8') as f:
                    data_sq = json.load(f)
                
                qas_list = data_sq.get("qas", [])
                
                if qas_list:
                    for qa in qas_list:
                        q_id = str(qa.get("question_id", ""))
                        if q_id:
                            # Chỉ lấy Question vào Input
                            result["input"][q_id] = qa.get("question", "")
                            # Bê nguyên si list Answers vào Output
                            result["output"][q_id] = qa.get("answers", [])
            except Exception as e:
                print(f"⚠️ Lỗi đọc SQuAD ({file_name}): {e}")
        
        # --- 5. Lưu file JSON ---
        output_file_path = path_task5 / file_name
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"✅ Hoàn tất! File sạch đã được lưu tại: {path_task5}")

if __name__ == "__main__":
    process_to_task5_refined()