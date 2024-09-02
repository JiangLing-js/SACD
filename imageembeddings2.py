import timm
import torch
from urllib.request import urlopen
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm

# 创建模型结构，但不加载预训练权重
base_model = timm.create_model('tf_efficientnet_b3_ns', pretrained=False, num_classes=0)

# 加载本地的预训练权重（假设为pytorch_model.bin）
checkpoint = torch.load('ckpt/pytorch_model.bin', map_location=torch.device('cuda'))

# 删除与模型不匹配的键
checkpoint.pop('classifier.weight', None)
checkpoint.pop('classifier.bias', None)

# 加载权重到模型
base_model.load_state_dict(checkpoint)

base_model = base_model.cuda()

# 设定训练参数
epoch_num = 3
bs_value = 32

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(base_model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# 定义要处理的文件夹列表
folders = [
    'ai_test', 'biggan_test', 'glide_test', 'ilsvrc',
    'ilsvrc_adm', 'nature_test', 'sdv4_test', 'sdv5_test',
    'VQDM_test', 'wukong_test'
]

# 创建保存特征的文件夹
output_folder = Path('tf_features')
output_folder.mkdir(exist_ok=True)

# 定义允许处理的图像扩展名
valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# 循环处理每个文件夹
for folder in folders:
    image_folder = Path(folder)

    # 创建一个字典来保存 embeddings
    embeddings_dict = {}

    # 循环处理文件夹中的每个图像
    for img_path in tqdm(image_folder.glob('*.*'), desc=f'Processing {folder}'):  # 使用tqdm显示进度
        if img_path.suffix.lower() not in valid_extensions:
            # 跳过非图像文件
            print(f'Skipping non-image file: {img_path}')
            continue

        try:
            img = Image.open(img_path)

            # 检查并转换为 RGB 图像
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 应用变换
            transformed_img = transforms(img).unsqueeze(0)  # 添加 batch 维度

            # 前向传递
            output = base_model.forward_features(transformed_img.cuda())

            # 如果需要，可以对输出进行进一步处理
            pooled_output = base_model.forward_head(output, pre_logits=True)

            # 将 embedding 保存到字典中
            embeddings_dict[img_path.name] = pooled_output.cpu().detach().numpy()  # 转换为 numpy 数组
        except Exception as e:
            print(f'Error processing {img_path}: {e}')
            continue

    # 保存 embeddings 到 .pth 文件
    output_path = output_folder / f'{folder}_features.pth'
    torch.save(embeddings_dict, output_path)

    print(f'Embeddings saved to {output_path}')
