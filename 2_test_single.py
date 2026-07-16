import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision.ops import nms
import tempfile
import os

# ==================== 配置区 ====================
SLICE_SIZE = 416  # 切片尺寸（与训练一致）
OVERLAP = 0.3  # 重叠比例 30%
CONF_THRESHOLD = 0.25  # 置信度阈值
IOU_THRESHOLD = 0.5  # NMS去重阈值
LARGE_IMAGE_THRESHOLD = 1024  # 图片宽或高超过此值，触发切片模式
# ===============================================

# 1. 配置网页标题和图标
st.set_page_config(page_title="树木智能检测系统", page_icon="🌲")
st.title("🌲 树木目标检测系统 (YOLOv8)")
st.write("上传一张照片，AI 将自动为你框选出里面的树木。")

# 2. 载入模型
model_path = "runs/runs/disease_tree_train/weights/best.pt"


@st.cache_resource
def load_my_model():
    return YOLO(model_path)


try:
    model = load_my_model()
    class_names = model.names
    st.sidebar.success(f"✅ 成功载入模型！类别: {class_names}")
except Exception as e:
    st.sidebar.error(f"❌ 模型载入失败，请检查路径。错误: {e}")
    st.stop()


# ==================== 切片推理核心函数 ====================

def load_image_to_numpy(uploaded_file):
    """从上传文件读取图片，统一转为 RGB numpy 数组"""
    img = Image.open(uploaded_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def slice_and_detect(image_np, model, slice_size=416, overlap=0.3, conf_threshold=0.25):
    """
    对大图进行切片推理，返回所有检测框（映射回原图坐标）
    """
    h, w = image_np.shape[:2]
    step = int(slice_size * (1 - overlap))

    all_boxes = []

    # 计算切片数量
    y_steps = max(1, (h - slice_size + step - 1) // step + 1)
    x_steps = max(1, (w - slice_size + step - 1) // step + 1)
    total_slices = y_steps * x_steps

    # 显示进度（用 Streamlit 的进度条）
    progress_bar = st.progress(0, text="🔪 正在切片推理...")
    status_text = st.empty()

    processed = 0
    for y in range(0, h - slice_size + 1, step):
        for x in range(0, w - slice_size + 1, step):
            y_start = min(y, h - slice_size)
            x_start = min(x, w - slice_size)

            slice_img = image_np[y_start:y_start + slice_size, x_start:x_start + slice_size]

            results = model(slice_img, conf=conf_threshold, verbose=False)

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xywh.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confs, cls_ids):
                    cx_rel, cy_rel, bw, bh = box
                    cx_abs = x_start + cx_rel
                    cy_abs = y_start + cy_rel
                    all_boxes.append({
                        'xywh': [cx_abs, cy_abs, bw, bh],
                        'conf': float(conf),
                        'cls': int(cls_id)
                    })

            processed += 1
            progress = processed / total_slices
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"处理切片: {processed}/{total_slices}")

    status_text.text(f"✅ 切片完成，共检测到 {len(all_boxes)} 个候选框")
    progress_bar.empty()

    return all_boxes


def apply_nms(boxes_list, iou_threshold=0.5):
    """NMS 去重"""
    if len(boxes_list) == 0:
        return []

    boxes_np = np.array([b['xywh'] for b in boxes_list])
    confs_np = np.array([b['conf'] for b in boxes_list])

    x1 = boxes_np[:, 0] - boxes_np[:, 2] / 2
    y1 = boxes_np[:, 1] - boxes_np[:, 3] / 2
    x2 = boxes_np[:, 0] + boxes_np[:, 2] / 2
    y2 = boxes_np[:, 1] + boxes_np[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep_indices = nms(
        torch.tensor(boxes_xyxy, dtype=torch.float32),
        torch.tensor(confs_np, dtype=torch.float32),
        iou_threshold=iou_threshold
    ).numpy()

    return [boxes_list[i] for i in keep_indices]


def draw_boxes(image_np, boxes, class_names):
    """绘制检测框"""
    img_copy = image_np.copy()

    for box in boxes:
        cx, cy, w, h = box['xywh']
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        # 病树(类别1)用红色，死树(类别0)用蓝色
        color = (0, 0, 255) if box['cls'] == 1 else (255, 0, 0)
        label = f"{class_names[box['cls']]} {box['conf']:.2f}"

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_copy, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img_copy


def process_image(image_np, model, class_names):
    """
    智能处理：小图直接识别，大图切片推理
    返回 (结果图片, 检测框列表, 模式说明)
    """
    h, w = image_np.shape[:2]

    # 判断是否需要切片
    if h > LARGE_IMAGE_THRESHOLD or w > LARGE_IMAGE_THRESHOLD:
        mode = "切片推理（大图模式）"
        st.info(f"📐 图片尺寸 {w}x{h} 超过阈值，启用切片推理（切片尺寸 {SLICE_SIZE}）")

        all_boxes = slice_and_detect(
            image_np, model,
            slice_size=SLICE_SIZE,
            overlap=OVERLAP,
            conf_threshold=CONF_THRESHOLD
        )
        final_boxes = apply_nms(all_boxes, iou_threshold=IOU_THRESHOLD)
        result_img = draw_boxes(image_np, final_boxes, class_names)

        return result_img, final_boxes, mode
    else:
        mode = "直接推理（常规模式）"
        st.info(f"📐 图片尺寸 {w}x{h}，直接推理")

        results = model(image_np, conf=CONF_THRESHOLD, verbose=False)
        result_img = results[0].plot()

        # 提取检测框信息
        boxes_list = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                boxes_list.append({
                    'xywh': box.tolist(),
                    'conf': float(conf),
                    'cls': int(cls_id)
                })

        return result_img, boxes_list, mode


# ==================== Streamlit UI ====================

uploaded_file = st.file_uploader(
    "请选择一张树木图片...",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"]
)

if uploaded_file is not None:
    # 读取图片
    with st.spinner("📂 正在加载图片..."):
        image_np = load_image_to_numpy(uploaded_file)

    # 显示原图尺寸
    h, w = image_np.shape[:2]
    st.sidebar.info(f"📷 原图尺寸: {w} x {h}")

    # 界面分两栏
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 原始图片")
        st.image(image_np, use_container_width=True)

    with col2:
        st.subheader("🤖 AI 检测结果")
        with st.spinner("AI 正在识别中..."):
            result_img, boxes, mode = process_image(image_np, model, class_names)

        # 显示检测结果
        st.image(result_img, channels="BGR", use_container_width=True)
        st.success(f"✅ 检测完成！模式: {mode}")

    # 显示检测目标详细信息
    st.subheader("📊 检测目标列表")

    if len(boxes) == 0:
        st.info("未检测到任何目标")
    else:
        # 用表格展示
        data = []
        for idx, box in enumerate(boxes, 1):
            cls_name = class_names.get(box['cls'], f"class_{box['cls']}")
            cx, cy, w_box, h_box = box['xywh']
            data.append({
                "序号": idx,
                "类别": cls_name,
                "置信度": f"{box['conf']:.3f}",
                "中心X": f"{cx:.1f}",
                "中心Y": f"{cy:.1f}",
                "宽度": f"{w_box:.1f}",
                "高度": f"{h_box:.1f}"
            })
        st.dataframe(data, use_container_width=True)

        # 统计信息
        st.sidebar.success(f"🎯 共检测到 {len(boxes)} 个目标")

        # 按类别统计
        from collections import Counter

        cls_counts = Counter([class_names.get(b['cls'], f"class_{b['cls']}") for b in boxes])
        st.sidebar.write("📊 类别统计:")
        for cls_name, count in cls_counts.items():
            st.sidebar.write(f"  - {cls_name}: {count} 个")

            #   streamlit run 2_test_single.py