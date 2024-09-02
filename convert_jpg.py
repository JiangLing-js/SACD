import os
from PIL import Image
import subprocess

# 设置原始图像路径和输出路径
original_image_folder = ""
output_image_folder = "output-image"


# 确保输出文件夹存在
os.makedirs(output_image_folder, exist_ok=True)

# 数据集列表
datasets = ["ai_test", "biggan_test", "glide_test", "ilsvrc", "ilsvrc_adm", "nature_test", "SAN_test", "sdv4_test", "sdv5_test", "VQDM_test", "wukong_test"]
quality_levels = [95, 85, 75]

for dataset in datasets:
    dataset_path = os.path.join(original_image_folder, dataset)

    for quality in quality_levels:
        quality_folder = f"quality_{quality}"
        output_quality_folder = os.path.join(output_image_folder, quality_folder, dataset)
        os.makedirs(output_quality_folder, exist_ok=True)

        for image_name in os.listdir(dataset_path):
            if image_name.lower().endswith(".png"):
                image_path = os.path.join(dataset_path, image_name)
                image = Image.open(image_path)

                # 如果图像是RGBA模式，转换为RGB模式
                if image.mode == 'RGBA':
                    image = image.convert('RGB')

                base_name = os.path.splitext(image_name)[0]
                jpg_image_path = os.path.join(output_quality_folder, f"{base_name}.jpg")
                image.save(jpg_image_path, "JPEG", quality=quality)

print("All images processed and saved with different JPEG quality levels.")
