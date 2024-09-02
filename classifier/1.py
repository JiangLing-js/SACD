from contrastive_classifier import FeatureClassifierModel
import torch

# 实例化模型
model = FeatureClassifierModel(num_channels=256, num_classes=2)

# 加载模型权重
model.load_state_dict(torch.load('model_weights.pth'))

# 计算并打印模型的可学习参数总数
total_params = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        num_params = param.numel()
        total_params += num_params
        print(f"Parameter: {name}, Size: {param.size()}, Number of Elements: {num_params}")

print(f"Total number of learnable parameters: {total_params}")
