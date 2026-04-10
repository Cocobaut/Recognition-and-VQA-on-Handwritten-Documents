# Sử dụng base image có sẵn Python 3.10, PyTorch và CUDA (Hỗ trợ GPU)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Thiết lập thư mục làm việc mặc định bên trong Container
WORKDIR /app

# Khắc phục lỗi timezone nếu có thư viện nào yêu cầu
ENV TZ=Asia/Ho_Chi_Minh
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Cài đặt các công cụ cơ bản của Linux (phòng trường hợp cần thiết)
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements.txt vào container
COPY docker/requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Lệnh mặc định khi container khởi chạy
CMD ["/bin/bash"]