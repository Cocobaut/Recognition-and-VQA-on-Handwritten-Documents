import json
import os
import math
from pathlib import Path
from tqdm import tqdm
import Config.config as config
import cv2
import numpy as np

# 1. Cấu hình
Task_2_Config = config.return_Task2_Predict_Config()
dataset_input_dir = Task_2_Config["input_json"]
output_local_dir = Task_2_Config["output_json"]
image_dir = Task_2_Config["input_images"]

os.makedirs(output_local_dir, exist_ok=True)

def get_obb_properties(p):
    """Tính toán các thuộc tính của hộp xoay từ 4 đỉnh polygon"""

    y_coords = [p['y0'], p['y1'], p['y2'], p['y3']]
    
    y_min = min(y_coords)
    y_max = max(y_coords)

    # Tâm Y là trung bình cộng của 4 tọa độ y
    y_center = sum(y_coords) / 4
    
    # Tâm X là trung bình cộng của 4 tọa độ x
    x_center = (p['x0'] + p['x1'] + p['x2'] + p['x3']) / 4
    
    # Chiều cao hộp (khoảng cách giữa cạnh trên và cạnh dưới) - (Tính trung bình độ dài 2 cạnh bên: (P0-P3) và (P1-P2))
    h = (math.sqrt((p['x3'] - p['x0'])**2 + (p['y3'] - p['y0'])**2) +   
         math.sqrt((p['x2'] - p['x1'])**2 + (p['y2'] - p['y1'])**2)) / 2
    
    # Góc nghiêng (radian) dựa trên cạnh ngang trên (P0-P1)
    angle = math.atan2(p['y1'] - p['y0'], p['x1'] - p['x0'])
    
    return y_center, y_min, y_max, x_center, h, angle

def get_area(p):
    """Tính diện tích hình chữ nhật từ polygon (x0,y0...x3,y3)"""
    width = math.sqrt((p['x1'] - p['x0'])**2 + (p['y1'] - p['y0'])**2)
    height = math.sqrt((p['x3'] - p['x0'])**2 + (p['y3'] - p['y0'])**2)
    
    return max(1.0, width * height)

def get_intersection_area(box1, box2):
    """Tính diện tích phần giao nhau giữa 2 box (AABB approximation)"""
    x_left = max(box1['x0'], box2['x0'])
    y_top = max(box1['y0'], box2['y0'])
    x_right = min(box1['x2'], box2['x2'])
    y_bottom = min(box1['y2'], box2['y2'])

    if x_right < x_left or y_bottom < y_top: 
        return 0.0
    
    return (x_right - x_left) * (y_bottom - y_top)

def refine_overlapping_lines(lines, overlap_threshold=0.85):
    if not lines: 
        return []
    
    lines.sort(key=lambda x: get_area(x['polygon']), reverse=True)

    refined = []
    skip_indices = set()

    for i in range(len(lines)):
        if i in skip_indices: 
            continue
        
        curr_line = lines[i]
        
        for j in range(i + 1, len(lines)):
            if j in skip_indices: 
                continue
            
            other_line = lines[j]
            inter_area = get_intersection_area(curr_line['polygon'], other_line['polygon'])
            area_other = get_area(other_line['polygon'])
            
            # Nếu phần giao chiếm > 60% diện tích của box nhỏ hơn -> Gộp
            if area_other > 0 and (inter_area / area_other) > overlap_threshold:
                curr_line['word_ids'] = list(set(curr_line['word_ids'] + other_line['word_ids']))
                p1, p2 = curr_line['polygon'], other_line['polygon']
                curr_line['polygon'] = {
                    "x0": min(p1['x0'], p2['x0']), "y0": min(p1['y0'], p2['y0']),
                    "x1": max(p1['x1'], p2['x1']), "y1": min(p1['y1'], p2['y1']),
                    "x2": max(p1['x2'], p2['x2']), "y2": max(p1['y2'], p2['y2']),
                    "x3": min(p1['x3'], p2['x3']), "y3": max(p1['y3'], p2['y3'])
                }
                skip_indices.add(j)

        refined.append(curr_line)
    
    refined.sort(key=lambda x: x['polygon']['y0'])
    
    for idx, r in enumerate(refined): 
        r['id'] = idx
    
    return refined

def final_grouping_pass(lines, img_width):
    """
    Bước cuối: Gom nhóm các dòng nếu chúng cùng hàng (Y-center) 
     và khoảng cách X không quá 1/5 chiều rộng ảnh.
    """
    if not lines: return []
    
    # Sắp xếp theo Y trước, X sau
    lines.sort(key=lambda l: ((l['polygon']['y0'] + l['polygon']['y2']) / 2, l['polygon']['x0']))
    
    refined = []
    skip_indices = set()

    for i in range(len(lines)):
        if i in skip_indices: continue
        curr = lines[i]
        
        for j in range(i + 1, len(lines)):
            if j in skip_indices: continue
            other = lines[j]
            
            # 1. Tính toán Y center và chiều cao dòng hiện tại
            y_c_curr = (curr['polygon']['y0'] + curr['polygon']['y2']) / 2
            y_c_other = (other['polygon']['y0'] + other['polygon']['y2']) / 2
            
            # Kiểm tra Y center không lệch quá 10 pixel
            v_match = abs(y_c_curr - y_c_other) <= 10
            
            # 2. Tính khoảng cách X giữa 2 bbox dòng
            # Tìm khoảng cách ngắn nhất giữa các biên X
            x_gap = min(abs(other['polygon']['x0'] - curr['polygon']['x1']), 
                        abs(curr['polygon']['x0'] - other['polygon']['x1']))
            
            h_match = x_gap < (img_width / 6)
            
            if v_match and h_match:
                # Tiến hành gộp
                curr['word_ids'] = list(set(curr['word_ids'] + other['word_ids']))
                p1, p2 = curr['polygon'], other['polygon']
                curr['polygon'] = {
                    "x0": min(p1['x0'], p2['x0']), "y0": min(p1['y0'], p2['y0']),
                    "x1": max(p1['x1'], p2['x1']), "y1": min(p1['y1'], p2['y1']),
                    "x2": max(p1['x2'], p2['x2']), "y2": max(p1['y2'], p2['y2']),
                    "x3": min(p1['x3'], p2['x3']), "y3": max(p1['y3'], p2['y3'])
                }
                skip_indices.add(j)
        
        refined.append(curr)
    
    # Đánh lại ID
    for idx, r in enumerate(refined): r['id'] = idx
    return refined

def check_connection_angle(w1_poly, w2_poly, max_angle_deg=20):
    """
    Kiểm tra góc tạo bởi đường nối tâm 2 từ.
    w1: Từ đã có trong dòng (thường là từ cuối cùng bên phải)
    w2: Từ mới đang xem xét gộp
    """
    # Tính tâm của 2 box
    c1_x = (w1_poly['x0'] + w1_poly['x1'] + w1_poly['x2'] + w1_poly['x3']) / 4
    c1_y = (w1_poly['y0'] + w1_poly['y1'] + w1_poly['y2'] + w1_poly['y3']) / 4
    
    c2_x = (w2_poly['x0'] + w2_poly['x1'] + w2_poly['x2'] + w2_poly['x3']) / 4
    c2_y = (w2_poly['y0'] + w2_poly['y1'] + w2_poly['y2'] + w2_poly['y3']) / 4
    
    # Tính góc của vector nối 2 tâm (tính bằng độ)
    dx = c2_x - c1_x
    dy = c2_y - c1_y
    
    if dx == 0: return False # Tránh chia cho 0 hoặc trùng X
    
    angle_rad = math.atan2(abs(dy), abs(dx))
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg < max_angle_deg

def grouping_logic_obb_pure(word_blocks,img_width, gap_threshold_ratio=3.5, angle_threshold=0.15, dual_overlap_thresh=0.6):
    if not word_blocks: return []
    
    processed_words = []
    for w in word_blocks:
        y_c, y_min, y_max, x_c, h, ang = get_obb_properties(w['polygon'])
        processed_words.append({
            'data': w, 'y_c': y_c, 'y_min': y_min, 'y_max': y_max, 'h': h, 'angle': ang
        })

    # Sắp xếp X để gom từ trái qua phải
    processed_words.sort(key=lambda x: x['data']['polygon']['x0'])
    lines_pool = [] 

    for w in processed_words:
        assigned = False
        w_poly = w['data']['polygon']
        
        for line in lines_pool:
            # Metadata dòng hiện tại
            l_y_min, l_y_max = line['y_min'], line['y_max']
            l_h_avg = line['h_sum'] / len(line['words'])
            last_w_in_line = line['words'][-1]
            
            # Điều kiện 1: Góc nghiêng tương đồng
            angle_match = abs(w['angle'] - last_w_in_line['angle']) < angle_threshold

            # Điều kiện 2: Khớp cao độ (Vertical Overlap)
            overlap = min(w['y_max'], l_y_max) - max(w['y_min'], l_y_min)   # Tính độ phủ dọc (Vertical Overlap)
            if overlap > 0:
                ratio_w = overlap / max(1, w['h'])                          # Tỉ lệ phủ so với từ mới (Ratio 1)
                ratio_l = overlap / max(1, l_h_avg)                         # Tỉ lệ phủ so với chiều cao trung bình của dòng (Ratio 2)
                vertical_match = (ratio_w > dual_overlap_thresh) and (ratio_l > dual_overlap_thresh)
            else:
                vertical_match = False
            
            # Điều kiện 3: Khoảng cách ngang
            horizontal_gap = w_poly['x0'] - last_w_in_line['data']['polygon']['x1']
            horizontal_match = horizontal_gap < (l_h_avg * gap_threshold_ratio)
            
            # Điều kiện 4: Kiểm tra xem từ/dòng mới có nằm trên 'đường bay' của dòng hiện tại không
            slope_match = check_connection_angle(last_w_in_line['data']['polygon'], w_poly, max_angle_deg=25)

            if angle_match and vertical_match and horizontal_match and slope_match:
                line['words'].append(w)
                line['y_min'] = min(l_y_min, w['y_min'])
                line['y_max'] = max(l_y_max, w['y_max'])
                line['h_sum'] += w['h']
                assigned = True
                break
        
        if not assigned:
            lines_pool.append({
                'words': [w],
                'y_min': w['y_min'],
                'y_max': w['y_max'],
                'h_sum': w['h']
            })

    # Tạo Output JSON tạm thời từ các nhóm đã gom
    initial_lines = []
    for i, lp in enumerate(lines_pool):
        line_words = lp['words']
        
        # Lấy TẤT CẢ tọa độ x và y từ cả 4 đỉnh (x0, x1, x2, x3)
        all_x = []
        all_y = []
        for lw in line_words:
            poly = lw['data']['polygon']
            all_x.extend([poly['x0'], poly['x1'], poly['x2'], poly['x3']])
            all_y.extend([poly['y0'], poly['y1'], poly['y2'], poly['y3']])
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        initial_lines.append({
            "id": i,
            "polygon": {
                "x0": min_x, "y0": min_y,
                "x1": max_x, "y1": min_y,
                "x2": max_x, "y2": max_y,
                "x3": min_x, "y3": max_y
            },
            "word_ids": [lw['data']['id'] for lw in line_words]
        })

    # Bước 1: Xử lý các hộp lồng nhau (Refine diện tích)
    first_lines = refine_overlapping_lines(initial_lines, overlap_threshold=0.65)
    
    # Bước 2: Gom nhóm (Y-center & 1/5 Image Width)
    second_lines = final_grouping_pass(first_lines, img_width)

    #Bước 3: Xử lí các hộp lồng nhau lần cuối theo diện tích
    final_lines = refine_overlapping_lines(second_lines, overlap_threshold=0.65)

    return final_lines

def find_image_path(image_dir, file_stem):
    """
    Tìm đường dẫn ảnh với các đuôi mở rộng phổ biến.
    """
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']
    for ext in extensions:
        path = os.path.join(image_dir, file_stem + ext)
        if os.path.exists(path):
            return path
    return None

def imread_unicode(path):
    """
    Đọc ảnh từ đường dẫn chứa ký tự Unicode (có dấu).
    """
    try:
        img_array = np.fromfile(path, np.uint8)             # Sử dụng numpy để đọc file dưới dạng mảng byte
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)     # Decode mảng byte đó thành định dạng ảnh của OpenCV
        return img
    except Exception as e:
        return None

def run_verification():
    json_files = list(Path(dataset_input_dir).glob('*.json'))
    if not json_files:
        print(f"- Không tìm thấy file JSON nào tại: {dataset_input_dir}")
        return

    print(f"- Đang xử lý gom nhóm dòng cho {len(json_files)} file...")
    for j_path in tqdm(json_files):
        # Lấy kích thước ảnh để tính 1/5 width
        img_path = find_image_path(image_dir, j_path.stem)
        
        if img_path is not None:
            img = imread_unicode(img_path)
            if img is not None:
                img_width = img.shape[1]
            else:
                img_width = 2000    # Trường hợp file lỗi/hỏng
        else:
            img_width = 2000        # Nếu không tìm thấy bất kỳ file ảnh nào trùng tên

        with open(j_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        input_words = data.get('input', {}).get('text_blocks', [])
        if not input_words: input_words = data.get('output', {}).get('text_blocks', [])
            
        computed_output = grouping_logic_obb_pure(input_words, img_width)
        
        result_json = {
            "task_info": {"name": "Task 2 Final", "total_lines": len(computed_output)},
            "input": {"text_blocks": input_words},
            "output_predicted": {"text_blocks": computed_output}
        }
        with open(os.path.join(output_local_dir, j_path.name), 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    run_verification()