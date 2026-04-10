from Config import config

Task_3_Predict_Config = config.return_Task3_Predict_Config()

# Cấu hình file đường dẫn 
json_dir = Task_3_Predict_Config["input_json"]
image_dir = Task_3_Predict_Config["input_images"]

# 1. Cấu hình thiết bị & Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = Task_3_Predict_Config["weight"]

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
model.eval()

def generate_task3_json():
    output_dir = Task_3_Predict_Config["output_json"]
    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    already_done = len([f for f in os.listdir(output_dir) if f.endswith('.json')])
    print(f"- Tổng cộng {len(json_files)} file. Đã xử lý xong {already_done} file.")
    print(f"- Đang tiếp tục xử lý {len(json_files) - already_done} file còn lại...")

    for json_filename in tqdm(json_files):
        output_json_path = os.path.join(output_dir, json_filename)
        
        if os.path.exists(output_json_path):
            continue

        json_path = os.path.join(json_dir, json_filename)
        base_name = os.path.splitext(json_filename)[0]

        image_path = None
        for ext in valid_extensions:
            temp_path = os.path.join(image_dir, f"{base_name}{ext}")
            if os.path.exists(temp_path):
                image_path = temp_path
                break

        if image_path is None:
            continue

        image = Image.open(image_path).convert("RGB")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "output_predicted" in data and "text_blocks" in data["output_predicted"]:
            blocks_to_process = data["output_predicted"]["text_blocks"]
        else:
            blocks_to_process = data.get("input", {}).get("text_blocks", [])

        for block in blocks_to_process:
            polygon = block["polygon"]
            
            x_coords = [polygon["x0"], polygon["x1"], polygon["x2"], polygon["x3"]]
            y_coords = [polygon["y0"], polygon["y1"], polygon["y2"], polygon["y3"]]
            box = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

            cropped_image = image.crop(box)

            pixel_values = processor(cropped_image, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            block["text"] = pred_text
        
        # Lưu file ngay khi vừa chạy xong 1 ảnh
        with open(output_json_path, 'w', encoding='utf-8') as out_f:
            json.dump(data, out_f, ensure_ascii=False, indent=4)

    print(f"- Successfully. File Json is saved in: {output_dir}")

if __name__ == "__main__":
    generate_task3_json()