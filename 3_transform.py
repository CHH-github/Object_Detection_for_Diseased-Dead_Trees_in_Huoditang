import os
from PIL import Image
from pathlib import Path
#修改图片颜色通道由4到3通道

def convert_images_to_rgb(folder_path, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
    """
    将指定文件夹下所有图片转换为RGB模式（3通道）
    会直接覆盖原文件
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    # 统计信息
    converted_count = 0
    skipped_count = 0
    error_count = 0

    # 遍历文件夹下所有图片
    for file_path in folder.glob('*'):
        if file_path.suffix.lower() in extensions:
            try:
                img = Image.open(file_path)
                original_mode = img.mode

                # 只要不是RGB模式就转换
                if img.mode != 'RGB':
                    rgb_img = img.convert('RGB')
                    rgb_img.save(file_path, quality=95)
                    print(f"✅ 转换成功: {file_path.name} ({original_mode} -> RGB)")
                    converted_count += 1
                else:
                    print(f"⏭️ 已是RGB，跳过: {file_path.name}")
                    skipped_count += 1

            except Exception as e:
                print(f"❌ 处理失败: {file_path.name} - {e}")
                error_count += 1

    # 输出统计结果
    print("\n" + "=" * 50)
    print(f"📊 处理完成！")
    print(f"   ✅ 成功转换: {converted_count} 张")
    print(f"   ⏭️ 已是RGB: {skipped_count} 张")
    print(f"   ❌ 处理失败: {error_count} 张")
    print(f"   📁 总计扫描: {converted_count + skipped_count + error_count} 张")


if __name__ == "__main__":
    # ⚠️ 请修改为你要处理的文件夹路径
    target_folder = "images"

    # 调用转换函数
    convert_images_to_rgb(target_folder)