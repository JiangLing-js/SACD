from segment_anything import sam_model_registry, SamPredictor
import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

# 指定模型类型和检查点路径
model_type = "vit_b"  # 需要替换为实际的模型类型
ckpt_path = "ckpt/sam_vit_b_01ec64.pth"  # 需要替换为实际的路径

# 加载模型到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = sam_model_registry[model_type](checkpoint=ckpt_path).to(device)
predictor = SamPredictor(model)

# 定义图片转换，不调整大小
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 目标文件夹列表
folders = ['x2']

def process_folder(folder_path):
    features = {}  # 保存特征的字典
    # 获取文件夹中的图片文件列表
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    batch_size = 1  # 批处理的大小
    for i in tqdm(range(0, len(image_files), batch_size), desc=f"Processing {folder_path}"):
        batch_files = image_files[i:i + batch_size]
        images = [Image.open(os.path.join(folder_path, filename)).convert('RGB') for filename in batch_files]

        # 转换为Tensor并进行预处理
        image_tensors = torch.stack([transform(image) for image in images]).to(device)

        # 使用模型获取嵌入
        embeddings = []
        for image_tensor in image_tensors:
            predictor.set_image(image_tensor.permute(1, 2, 0).cpu().numpy())  # 转换为numpy数组
            embeddings.append(predictor.get_image_embedding())

        # 将嵌入保存到字典中
        for filename, embedding in zip(batch_files, embeddings):
            features[filename] = embedding

    # 保存特征到文件
    save_path = f'{folder_path}_features.pth'
    torch.save(features, save_path)
    print(f"Image features for {folder_path} saved successfully to {save_path}.")

# 处理所有目标文件夹
for folder in folders:
    process_folder(folder)
