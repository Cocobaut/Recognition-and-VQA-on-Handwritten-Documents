import os
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from PIL import Image

class FolderOCRDataset(Dataset):
    def __init__(self, image_dir, json_dir, processor):
        """
        image_dir: Thư mục chứa tất cả ảnh
        json_dir: Thư mục chứa tất cả file JSON tương ứng
        processor: TrOCRProcessor
        """
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.processor = processor

        # Quét thư mục và tạo danh sách toàn bộ các đoạn text (samples)
        self.samples = self._prepare_all_samples()

    def _prepare_all_samples(self):
        all_samples = []

        # Lấy danh sách tất cả các file JSON trong thư mục
        json_files = sorted([f for f in os.listdir(self.json_dir) if f.endswith('.json')])

        # Định nghĩa các đuôi ảnh cho phép
        valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

        for json_filename in json_files:
            json_path = os.path.join(self.json_dir, json_filename)
            base_name = os.path.splitext(json_filename)[0]

            image_path = None

            # Lặp qua các đuôi mở rộng để tìm ảnh tồn tại
            for ext in valid_extensions:
                temp_path = os.path.join(self.image_dir, f"{base_name}{ext}")
                if os.path.exists(temp_path):
                    image_path = temp_path
                    break # Dừng tìm kiếm khi đã thấy ảnh

            # Bỏ qua nếu không tìm thấy bất kỳ ảnh nào tương ứng với JSON
            if image_path is None:
                print(f"Cảnh báo: Không tìm thấy ảnh hợp lệ (.jpg, .jpeg, .png) cho {json_filename}")
                continue

            # Đọc file JSON hiện tại
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            input_blocks = data.get("input", {}).get("text_blocks", [])
            output_blocks = data.get("output", {}).get("text_blocks", [])   

            id_to_text = {block["id"]: block["text"] for block in output_blocks}

            # Xử lý từng block trong file JSON hiện tại
            for block in input_blocks:
                block_id = block["id"]
                if block_id in id_to_text:
                    polygon = block["polygon"]
                    x_coords = [polygon["x0"], polygon["x1"], polygon["x2"], polygon["x3"]]
                    y_coords = [polygon["y0"], polygon["y1"], polygon["y2"], polygon["y3"]]

                    bounding_box = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

                    all_samples.append({
                        "image_path": image_path,
                        "box": bounding_box,
                        "text": id_to_text[block_id]
                    })

        return all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Lấy thông tin của 1 sample cụ thể
        sample = self.samples[idx]
        image_path = sample["image_path"]
        box = sample["box"]
        text = sample["text"]

        # Cơ chế lazy-loading
        image = Image.open(image_path).convert("RGB")
        cropped_image = image.crop(box)

        pixel_values = self.processor(cropped_image, return_tensors="pt").pixel_values.squeeze()

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=128,
            truncation=True
        ).input_ids

        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}