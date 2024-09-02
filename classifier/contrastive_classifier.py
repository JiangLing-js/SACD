import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureClassifierModel(nn.Module):
    def __init__(self, num_channels, num_classes=2):
        super(FeatureClassifierModel, self).__init__()
        self.conv = nn.Conv2d(256, 256, kernel_size=(64, 64), stride=1, padding=0, groups=256)
        self.fc1 = nn.Linear(1536, 256)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)  # 展平特征
        z = F.normalize(x, p=2, dim=0)  # L2归一化得到z
        logits = self.classifier(x)
        return z, logits
        # x = F.relu(self.fc1(x))  # 通过第一层全连接层并应用ReLU激活
        # z = F.normalize(x, p=2, dim=1)  # L2归一化得到z
        # logits = self.classifier(x)  # 通过第二层全连接层得到logits
        # return z, logits

def psc_loss(zs, labels, prototypes, tau=1.0):
    # z: 样本的特征向量
    # label: 样本的类标签
    # prototypes: [num_classes, feature_dim] 每个类的原型
    # tau: 温度参数
    # 计算所有样本与所有原型之间的点积
    logits = torch.mm(zs, prototypes.t()) / tau
    nums = len(zs)
    # 使用 labels 来获取每个样本与其对应原型的 logits
    loss = 0
    for i in range(nums):
        logit_z = logits[i][labels[i]]
        logit_o = logits[i][1-labels[i]]
        loss += -torch.log(torch.exp(logit_z)/torch.exp(logit_o))

    return loss

# 计算类原型，这应当基于当前的数据集或一个批次的平均
def compute_prototype(zs, labels):
    # zs is a list of tensors, each tensor is a feature vector of a sample
    # labels is a list of integers indicating the class label of each sample in zs

    # Create lists to store tensors by class
    class_0 = [z for z, label in zip(zs, labels) if label == 0]
    class_1 = [z for z, label in zip(zs, labels) if label == 1]

    # Calculate the prototype for each class
    prototype_0 = torch.mean(torch.stack(class_0), dim=0) if class_0 else None
    prototype_1 = torch.mean(torch.stack(class_1), dim=0) if class_1 else None

    # Stack prototypes into a single tensor
    # Note: We must check if prototypes are None (in case there are no elements for a class)
    if prototype_0 is not None and prototype_1 is not None:
        prototypes = torch.stack([prototype_0, prototype_1])
    elif prototype_0 is not None:
        prototypes = prototype_0.unsqueeze(0)  # Only class 0 has samples
    elif prototype_1 is not None:
        prototypes = prototype_1.unsqueeze(0)  # Only class 1 has samples
    else:
        prototypes = torch.tensor([])  # No samples available

    return prototypes
