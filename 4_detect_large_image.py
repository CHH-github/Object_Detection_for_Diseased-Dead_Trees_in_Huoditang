#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大图切片推理脚本
直接修改下方 INPUT_IMAGE 和 OUTPUT_IMAGE 路径，然后运行：
    python detect_large_image.py
"""

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision.ops import nms
import torch
import time
import os

# ==================== ✏️ 在这里修改路径 ====================
INPUT_IMAGE = "input_images/dom_screennail.png"   # 要检测的图片
OUTPUT_IMAGE = "output/result.jpg" # 输出结果
MODEL_PATH = "runs/runs/disease_tree_train/weights/best.pt"                    # 模型路径
# ============================================================

# ==================== 检测参数 ====================
SLICE_SIZE = 416          # 切片尺寸（与训练一致）
OVERLAP = 0.3             # 重叠比例
CONF_THRESHOLD = 0.25     # 置信度阈值
IOU_THRESHOLD = 0.5       # NMS 去重阈值
# =================================================


def load_image(image_path):
    """加载图片，统一转为 RGB numpy 数组"""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def slice_and_detect(image_np, model):
    """切片推理，返回映射到原图坐标的所有检测框"""
    h, w = image_np.shape[:2]
    step = int(SLICE_SIZE * (1 - OVERLAP))
    all_boxes = []

    print(f"📐 原图尺寸: {w} x {h}")
    print(f"🔪 切片尺寸: {SLICE_SIZE}, 重叠: {OVERLAP*100:.0f}%")

    total_slices = ((h - SLICE_SIZE) // step + 1) * ((w - SLICE_SIZE) // step + 1)
    count = 0

    for y in range(0, h - SLICE_SIZE + 1, step):
        for x in range(0, w - SLICE_SIZE + 1, step):
            y_start = min(y, h - SLICE_SIZE)
            x_start = min(x, w - SLICE_SIZE)

            slice_img = image_np[y_start:y_start + SLICE_SIZE, x_start:x_start + SLICE_SIZE]
            results = model(slice_img, conf=CONF_THRESHOLD, verbose=False)

            if results[0].boxes is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confs, cls_ids):
                    cx_rel, cy_rel, bw, bh = box
                    all_boxes.append({
                        'xywh': [x_start + cx_rel, y_start + cy_rel, bw, bh],
                        'conf': float(conf),
                        'cls': int(cls_id)
                    })

            count += 1
            if count % 10 == 0:
                print(f"  进度: {count}/{total_slices} 切片")

    print(f"✅ 切片完成，检测到 {len(all_boxes)} 个候选框")
    return all_boxes


def apply_nms(boxes_list):
    """NMS 去重"""
    if not boxes_list:
        return []

    boxes_np = np.array([b['xywh'] for b in boxes_list])
    confs_np = np.array([b['conf'] for b in boxes_list])

    x1 = boxes_np[:, 0] - boxes_np[:, 2] / 2
    y1 = boxes_np[:, 1] - boxes_np[:, 3] / 2
    x2 = boxes_np[:, 0] + boxes_np[:, 2] / 2
    y2 = boxes_np[:, 1] + boxes_np[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(
        torch.tensor(boxes_xyxy, dtype=torch.float32),
        torch.tensor(confs_np, dtype=torch.float32),
        IOU_THRESHOLD
    ).numpy()

    return [boxes_list[i] for i in keep]


def draw_boxes(image_np, boxes, class_names):
    """绘制检测框"""
    img = image_np.copy()
    for box in boxes:
        cx, cy, w, h = box['xywh']
        x1, y1 = int(cx - w/2), int(cy - h/2)
        x2, y2 = int(cx + w/2), int(cy + h/2)

        color = (0, 0, 255) if box['cls'] == 1 else (255, 0, 0)
        label = f"{class_names[box['cls']]} {box['conf']:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def main():
    print("="*60)
    print("🌳 树木检测系统 - 大图切片推理")
    print("="*60)

    # 检查输入文件
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ 输入图片不存在: {INPUT_IMAGE}")
        print("请修改脚本开头的 INPUT_IMAGE 变量")
        return

    # 检查模型
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型不存在: {MODEL_PATH}")
        print("请修改脚本开头的 MODEL_PATH 变量")
        return

    print(f"📂 输入: {INPUT_IMAGE}")
    print(f"📦 模型: {MODEL_PATH}")

    # 加载模型
    model = YOLO(MODEL_PATH)
    class_names = model.names
    print(f"✅ 模型加载成功，类别: {class_names}")

    # 加载图片
    image_np = load_image(INPUT_IMAGE)
    h, w = image_np.shape[:2]
    print(f"📐 图片尺寸: {w} x {h}")

    # 切片推理
    print("\n🔪 开始切片推理...")
    start = time.time()
    all_boxes = slice_and_detect(image_np, model)

    # NMS 去重
    print(f"\n🔄 NMS 去重 (IOU阈值: {IOU_THRESHOLD})...")
    final_boxes = apply_nms(all_boxes)
    print(f"✅ 去重完成: {len(all_boxes)} -> {len(final_boxes)} 个目标")

    # 绘制并保存
    print("\n🎨 绘制检测结果...")
    result_img = draw_boxes(image_np, final_boxes, class_names)
    cv2.imwrite(OUTPUT_IMAGE, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print(f"💾 结果已保存: {OUTPUT_IMAGE}")
    print(f"⏱️  总耗时: {time.time() - start:.2f} 秒")

    # 打印检测详情
    if final_boxes:
        print("\n" + "="*60)
        print("📊 检测目标列表")
        print("="*60)
        print(f"{'序号':<6} {'类别':<12} {'置信度':<10} {'中心X':<10} {'中心Y':<10} {'宽度':<10} {'高度':<10}")
        print("-"*60)
        for idx, b in enumerate(final_boxes, 1):
            name = class_names[b['cls']]
            cx, cy, w_box, h_box = b['xywh']
            print(f"{idx:<6} {name:<12} {b['conf']:<10.3f} {cx:<10.1f} {cy:<10.1f} {w_box:<10.1f} {h_box:<10.1f}")
        print("="*60)
        print(f"🎯 共检测到 {len(final_boxes)} 个目标")
    else:
        print("⚠️  未检测到任何目标")

    print("="*60)


if __name__ == "__main__":
    main()