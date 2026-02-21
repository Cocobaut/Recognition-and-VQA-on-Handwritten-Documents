import json
import os
from pathlib import Path
from tqdm import tqdm
import Config.config as config

Task_1_Train_Test_Config = config.return_Task1_Train_Test_Config()

def export_yolo_labels(json_dir, output_label_dir):
    os.makedirs(output_label_dir, exist_ok=True)
    json_files = list(Path(json_dir).glob('*.json'))
    
    print(f"-> Đang xuất nhãn sang {output_label_dir}...")
    for json_file in tqdm(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        yolo_lines = []
        for block in data['output']['text_blocks']:
            yolo = block.get('yolo_standard')
            if yolo:
                line = f"0 {yolo['x_center']} {yolo['y_center']} {yolo['w']} {yolo['h']}"
                yolo_lines.append(line)
        
        txt_path = Path(output_label_dir) / json_file.with_suffix('.txt').name
        with open(txt_path, 'w') as f:
            f.write("\n".join(yolo_lines))

# Thực hiện cho cả Train và Test
# Tạo thư mục labels
train_labels_path = Task_1_Train_Test_Config["input_images_train"].replace("images", "labels")
test_labels_path = Task_1_Train_Test_Config["input_images_test"].replace("images", "labels")

export_yolo_labels(Task_1_Train_Test_Config["json_train"], train_labels_path)
export_yolo_labels(Task_1_Train_Test_Config["json_test"], test_labels_path)
