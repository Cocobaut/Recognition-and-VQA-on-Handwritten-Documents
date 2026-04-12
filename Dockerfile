FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt thư viện hệ thống cần thiết cho việc xử lý ảnh (OpenCV)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements.txt và cài đặt
COPY requirements.txt .
# Cài đặt thêm các thư viện phục vụ API
RUN pip install --no-cache-dir fastapi uvicorn python-multipart gdown
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Thiết lập thư mục làm việc
WORKDIR /app

# CHỈ ĐỊNH CHO PYTHON BIẾT THƯ MỤC GỐC Ở ĐÂU
ENV PYTHONPATH="/app:/app/SubmissionFinalCode"

RUN python SubmissionFinalCode/Task1/Inference/set_up_model_task1.py
RUN python SubmissionFinalCode/Task3/Inference/set_up_model_task3.py
RUN python SubmissionFinalCode/Task4/Inference/set_up_model_task4.py
RUN python SubmissionFinalCode/Task5/Inference/set_up_model_task5.py

# Mở cổng 8000 cho API
EXPOSE 8000

# Lệnh khởi chạy API khi bật Container
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]