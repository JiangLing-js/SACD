from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import torch

# 假设你有一个有效的RGB图像作为Tensor
image_rgb = torch.randn(3, 224, 224)  # 示例数据，实际应用中应使用真实图像数据
print(image_rgb.shape)
