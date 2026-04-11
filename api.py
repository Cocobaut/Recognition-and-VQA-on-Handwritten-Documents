import os
import sys
import json
import subprocess
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
from contextlib import asynccontextmanager

import Config.config as pipeline_config

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Đang khởi động Server API...")
    print("✅ Hệ thống AI đã sẵn sàng!")
    yield
    print("🛑 Đang tắt Server...")

app = FastAPI(lifespan=lifespan, title="Handwritten VQA API")

@app.post("/predict/")
async def predict_vqa(
    image: UploadFile = File(...), 
    question: str = Form(...)
):
    try:
        t1_cfg = pipeline_config.return_Task1_Predict_Config()
        t5_cfg = pipeline_config.return_Task5_Predict_Config()
        
        input_img_dir = t1_cfg["input_images"]       
        input_text_dir = t5_cfg["input_text"]        
        output_json_final = t5_cfg["output_json"]    

        # SỬA LỖI TẠI ĐÂY: Chỉ tạo thư mục cha, không tạo thư mục trùng tên file
        os.makedirs(input_img_dir, exist_ok=True)
        os.makedirs(input_text_dir, exist_ok=True)
        os.makedirs(os.path.dirname(output_json_final), exist_ok=True)

        # Lưu ảnh
        img_path = os.path.join(input_img_dir, image.filename)
        with open(img_path, "wb") as buffer:
            buffer.write(await image.read())
            
        base_filename = os.path.splitext(image.filename)[0]
        
        # Lưu câu hỏi (Task 5 sẽ quét thư mục này và lấy câu hỏi mang mã 00001)
        question_data = {
                    "00001": question
        }
        question_path = os.path.join(input_text_dir, f"{base_filename}.json")
        with open(question_path, "w", encoding="utf-8") as f:
            json.dump(question_data, f, ensure_ascii=False, indent=4)

        # ---------------------------------------------------------
        # CHẠY AI PIPELINE
        # ---------------------------------------------------------
        print(f"⚙️ Đang xử lý ảnh: {image.filename}")
        my_env = os.environ.copy()
        my_env["PYTHONPATH"] = os.getcwd()
        try:
            subprocess.run([sys.executable, "SubmissionFinalCode/Task1/Inference/Task_1_predict.py"], env=my_env, check=True)
            subprocess.run([sys.executable, "SubmissionFinalCode/Task2/Inference/Task_2_predict.py"], env=my_env, check=True)
            subprocess.run([sys.executable, "SubmissionFinalCode/Task3/Inference/Task_3_predict.py"], env=my_env, check=True)
            subprocess.run([sys.executable, "SubmissionFinalCode/Task4/Inference/Task_4_predict.py"], env=my_env, check=True)
            subprocess.run([sys.executable, "SubmissionFinalCode/Task5/Inference/Task_5_predict.py"], env=my_env, check=True)
        except Exception as task_err:
            print(f"❌ Lỗi trong quá trình chạy Task: {task_err}")
            return {"status": "error", "message": "Lỗi khi chạy model AI."}

        # ---------------------------------------------------------
        # ĐỌC FILE KẾT QUẢ VÀ TRẢ VỀ API
        # ---------------------------------------------------------
        # SỬA LỖI TẠI ĐÂY: Đọc trực tiếp từ file kết quả do Task 5 nhả ra
        final_result_file = os.path.join(output_json_final, f"{base_filename}.json")
        answer = "Không tìm thấy đáp án"
        
        if os.path.exists(final_result_file):
            with open(final_result_file, "r", encoding="utf-8") as f:
                result_data = json.load(f)
                try:
                    # Lấy đáp án mang mã 00001 mà mình đã mớm cho AI ở trên
                    if isinstance(result_data, list):
                        answer = result_data[0]["predictions"]["00001"]["answer"]
                    else:
                        answer = result_data["00001"]["answer"]
                except KeyError:
                    answer = "Có đáp án nhưng không trích xuất được (Sai cấu trúc JSON)"

        # Dọn rác
        if os.path.exists(img_path): os.remove(img_path)
        if os.path.exists(question_path): os.remove(question_path)
        if os.path.exists(final_result_file): os.remove(final_result_file)

        return {
            "status": "success",
            "file": image.filename,
            "question": question,
            "answer": answer
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)