from ultralytics import YOLO
import torch

# 0. 确认GPU可用
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model = YOLO("yolov8n.pt")

# 2. 开始训练
results = model.train(
    # ===== 数据路径 =====
    data="disease_tree_recogniton.v1i.yolov8/data.yaml",  # 修改为你的data.yaml实际路径

    # ===== 核心参数（4GB显存专用） =====
    epochs=100,  # 训练轮数，100轮足够看出效果
    batch=8,  # 批次大小，4GB显存最大安全值就是8
    imgsz=416,  # ⚡ 关键：用416代替640，显存直接降一半

    # ===== 模型参数 =====
    device=0,  # 使用GPU

    # ===== 输出控制 =====
    project="../runs",  # 输出目录
    name="disease_tree_train",  # 实验名称
    exist_ok=True,  # 允许覆盖同名目录
    verbose=True,  # 打印详细训练日志

    # ===== 优化策略 =====
    patience=20,  # 20轮没提升就早停，节省时间
    cos_lr=True,  # 使用余弦退火学习率，训练更平滑
    amp=True,  # 混合精度训练，节省显存（这个对4GB非常关键！）

    # ===== 数据增强（小数据集必备） =====
    hsv_h=0.015,  # 色调增强
    hsv_s=0.7,  # 饱和度增强
    hsv_v=0.4,  # 明度增强
    degrees=10,  # 旋转±10度
    translate=0.1,  # 平移
    scale=0.5,  # 缩放
    shear=2,  # 剪切
    perspective=0.0,  # 透视（保持关闭）
    flipud=0.0,  # 上下翻转（保持关闭）
    fliplr=0.5,  # 左右翻转（病树左右翻转不影响判断）
    mosaic=1.0,  # Mosaic增强（有助于小目标）
    mixup=0.0,  # Mixup（数据少时不需要）
)

# 3. 训练完成后自动验证
print("\n=== 训练完成，开始验证 ===")
val_results = model.val()
print(f"mAP50: {val_results.box.map50:.4f}")
print(f"mAP50-95: {val_results.box.map:.4f}")