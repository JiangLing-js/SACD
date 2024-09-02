import os
from PIL import Image
import numpy as np
import cv2

# 设置原始图像路径和输出路径
original_image_folder = ""  # 请填入原始图像文件夹路径
output_image_folder = "output-image"

# 确保输出文件夹存在
os.makedirs(output_image_folder, exist_ok=True)

# 数据集列表
datasets = ["ai_test", "biggan_test", "glide_test", "ilsvrc", "ilsvrc_adm", "nature_test", "SAN_test", "sdv4_test",
            "sdv5_test", "VQDM_test", "wukong_test"]


def add_gaussian_noise(image, mean=0, var=0.01):
    """添加高斯噪声"""
    row, col, ch = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss, 0, 255)
    return noisy.astype(np.uint8)


def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    """添加椒盐噪声"""
    noisy = np.copy(image)
    total_pixels = image.size
    num_salt = np.ceil(salt_prob * total_pixels)
    num_pepper = np.ceil(pepper_prob * total_pixels)

    # 添加盐噪声
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 1

    # 添加椒噪声
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 0

    return noisy


def apply_median_filter(image, kernel_size=3):
    """应用中值滤波"""
    return cv2.medianBlur(image, kernel_size)


def process_image(image_path, output_folder, process_func, process_name):
    """处理单张图像，添加噪声或滤波，并保存为PNG图像"""
    try:
        with Image.open(image_path) as image:
            if image.mode == 'RGBA':
                image = image.convert('RGB')

            image_array = np.array(image)
            processed_image_array = process_func(image_array)
            processed_image_pil = Image.fromarray(processed_image_array)

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            processed_image_path = os.path.join(output_folder, f"{base_name}_{process_name}.png")
            processed_image_pil.save(processed_image_path, "PNG")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


for dataset in datasets:
    dataset_path = os.path.join(original_image_folder, dataset)

    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        continue

    # 创建处理后的输出文件夹
    noise_types = ["gaussian_noise", "salt_pepper_noise", "median_filter"]
    process_funcs = [add_gaussian_noise, add_salt_and_pepper_noise, apply_median_filter]

    for noise_type, process_func in zip(noise_types, process_funcs):
        output_noise_folder = os.path.join(output_image_folder, noise_type, dataset)
        os.makedirs(output_noise_folder, exist_ok=True)

        for image_name in os.listdir(dataset_path):
            if image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(dataset_path, image_name)

                if noise_type == "salt_pepper_noise":
                    process_image(image_path, output_noise_folder, lambda img: process_func(img, 0.01, 0.01),
                                  noise_type)
                else:
                    process_image(image_path, output_noise_folder, process_func, noise_type)

print("All images processed with noise and filters, and saved in respective folders.")
