import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import Config.config as config

# 1. Cấu hình
Task_2_Config = config.return_Task2_Predict_Config()
json_gt_dir = Task_2_Config["json_test"]
json_pred_dir = Task_2_Config["output_json_inference"]

def calculate_iou(poly1, poly2):
    """
    Tính Intersection over Union (IoU) giữa 2 polygon (dạng hình chữ nhật AABB)
    """
    # Tìm tọa độ bao quanh (min-max) cho poly1
    x_left_1, y_top_1 = poly1['x0'], poly1['y0']
    x_right_1, y_bottom_1 = poly1['x2'], poly1['y2']

    # Tìm tọa độ bao quanh (min-max) cho poly2
    x_left_2, y_top_2 = poly2['x0'], poly2['y0']
    x_right_2, y_bottom_2 = poly2['x2'], poly2['y2']

    # Tính toán tọa độ phần giao nhau
    x_inter_left = max(x_left_1, x_left_2)
    y_inter_top = max(y_top_1, y_top_2)
    x_inter_right = min(x_right_1, x_right_2)
    y_inter_bottom = min(y_bottom_1, y_bottom_2)

    if x_inter_right < x_inter_left or y_inter_bottom < y_inter_top:
        return 0.0

    inter_area = (x_inter_right - x_inter_left) * (y_inter_bottom - y_inter_top)
    area1 = (x_right_1 - x_left_1) * (y_bottom_1 - y_top_1)
    area2 = (x_right_2 - x_left_2) * (y_bottom_2 - y_top_2)
    
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def evaluate_grouping(gt_blocks, pred_blocks, iou_threshold=0.5):
    """
    Đánh giá dựa trên tọa độ Polygon (IoU) thay vì Word IDs
    """
    tp, fp, fn = 0, 0, 0
    gt_matched = [False] * len(gt_blocks)

    for p_block in pred_blocks:
        best_iou = 0
        best_gt_idx = -1
        
        for g_idx, g_block in enumerate(gt_blocks):
            if gt_matched[g_idx]: continue
            
            # So sánh dựa trên polygon thay vì word_ids để tránh KeyError
            iou = calculate_iou(p_block['polygon'], g_block['polygon'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
        
        if best_iou >= iou_threshold:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

    fn = len(gt_blocks) - tp
    return tp, fp, fn

def run_report():
    pred_files = list(Path(json_pred_dir).glob('*.json'))
    total_tp, total_fp, total_fn = 0, 0, 0
    
    print(f"- Đang đánh giá {len(pred_files)} file dựa trên Spatial IoU...")
    
    for p_path in tqdm(pred_files):
        gt_path = Path(json_gt_dir) / p_path.name
        if not gt_path.exists(): continue

        with open(p_path, 'r', encoding='utf-8') as f:
            # Predict có cấu trúc 'output_predicted'
            pred_data = json.load(f).get('output_predicted', {}).get('text_blocks', [])
        with open(gt_path, 'r', encoding='utf-8') as f:
            # GT có cấu trúc 'output'
            gt_data = json.load(f).get('output', {}).get('text_blocks', [])

        tp, fp, fn = evaluate_grouping(gt_data, pred_data)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*40)
    print(f"{'BÁO CÁO KẾT QUẢ TASK 2 (IoU)':^40}")
    print("="*40)
    print(f"Dòng mẫu (GT): {total_tp + total_fn}")
    print(f"Khớp đúng (TP): {total_tp}")
    print(f"Dư/Sai (FP):    {total_fp}")
    print(f"Bỏ sót (FN):   {total_fn}")
    print("-" * 40)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    run_report()