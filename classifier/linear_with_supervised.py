from contrastive_classifier import FeatureClassifierModel, psc_loss, compute_prototype
import torch
import torch.nn.functional as F
import os
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
import math
def load_and_pool_features(file_path):
    features = torch.load(file_path)
    return features

def prepare_data(features, label):
    X = []
    y = []
    for embedding in features.values():
        # Ensure embedding is reshaped or manipulated if necessary
        X.append(embedding[0])  # Removed the numpy conversion to keep as tensors
        y.append(label)
    return X, y

def prepare_and_convert_data(file_paths, labels):
    X, y = [], []
    for file_path, label in zip(file_paths, labels):
        features = load_and_pool_features(file_path)
        X_batch, y_batch = prepare_data(features, label)
        X.extend(X_batch)
        y.extend(y_batch)
    # Convert lists of tensors to a single tensor
    X_tensor = torch.stack(X)  # Use torch.stack to correctly form a batch tensor
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor


def prepare_test_data(file_paths, labels):
    X, y = [], []
    for file_path, label in zip(file_paths, labels):
        features = load_and_pool_features(file_path)
        X_batch, y_batch = prepare_data(features, label)
        # 将每个批次的列表直接转换为张量并存储
        X.append(torch.stack(X_batch))  # 使用 torch.stack 将列表转换为张量
        y.append(torch.tensor(y_batch, dtype=torch.long))
    # 如果需要在列表中保持张量，而不是转换为 NumPy 数组，则可以直接返回
    return X, y


# 特征文件路径和对应标签
# feature_files_labels = [
#     ('..\\features\\ai_test_features.pth', 1),
#     ('..\\features\\nature_test_features.pth', 0),
#     ('..\\features\\biggan_test_features.pth', 1),
#     ('..\\features\\wukong_test_features.pth', 1),
#     ('..\\features\\glide_test_features.pth', 1),
#     ('..\\features\\sdv4_test_features.pth', 1),
#     ('..\\features\\sdv5_test_features.pth', 1),
#     ('..\\features\\VQDM_test_features.pth', 1),
#     ('..\\features\\ilsvrc_features.pth', 0),
#     ('..\\features\\ilsvrc_adm_features.pth', 1),
#     ('..\\features\\SAN_test_features.pth', 1)
# ]

feature_files_labels = [
    ('..\\features\\ai_test_features.pth', 1),
    ('..\\features\\nature_test_features.pth', 0),
    ('..\\features\\biggan_test_features.pth', 1),
    ('..\\features\\wukong_test_features.pth', 1),
    ('..\\features\\glide_test_features.pth', 1),
    ('..\\features\\sdv4_test_features.pth', 1),
    ('..\\features\\sdv5_test_features.pth', 1),
    ('..\\features\\VQDM_test_features.pth', 1),
    ('..\\features\\ilsvrc_features.pth', 0),
    ('..\\features\\ilsvrc_adm_features.pth', 1),
    #('..\\features\\SAN_test_features.pth', 1)
]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 准备训练数据和测试数据
# train_files = ['..\\features\\ai_test_features.pth', '..\\features\\nature_test_features.pth']
# train_labels = [1, 0, 1]
train_files = ['..\\features\\ai_test_features.pth', '..\\features\\nature_test_features.pth']
train_labels = [1, 0]
X_train, y_train = prepare_and_convert_data(train_files, train_labels)
print(X_train.shape, y_train.shape)

X_train = X_train.to(device)
y_train = y_train.to(device)

# Dataset and DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# 训练线性分类器
model = FeatureClassifierModel(num_channels=256, num_classes=2).to(device)
# model.load_state_dict(torch.load('model_weights.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epochs = 26  # 根据需要调整
tau = 0.5
model.train()
alpha = 1
# 初始化全局最大值的移动平均
# global_max_loss_psc = 1.0
# global_max_loss_criterion = 1.0
# moving_average_decay = 0.9  # 移动平均的衰减系数
weight_psc_ = 1

for epoch in range(num_epochs):
    for images, labels in train_loader:
        # print(images.shape)
        # print(labels.shape)
        optimizer.zero_grad()
        z, outputs = model(images)
        # print(z.shape, outputs.shape)
        prototypes = compute_prototype(z, labels)
        # print(prototypes.shape)
        loss_psc = psc_loss(z, labels, prototypes, tau)
        loss_criterion = criterion(outputs, labels)
        # # 更新全局最大值的移动平均
        # global_max_loss_psc = moving_average_decay * global_max_loss_psc + (1 - moving_average_decay) * loss_psc.item()
        # global_max_loss_criterion = moving_average_decay * global_max_loss_criterion + (
        #             1 - moving_average_decay) * loss_criterion.item()
        # # 计算加权系数
        # ratio = math.fabs(loss_criterion.item() / loss_psc.item())
        # if ratio < 1:
        #     weight_factor = 10 ** (int(math.log10(ratio)))
        # else:
        #     weight_factor = 10 ** (-int(math.log10(1 / ratio)))
        #
        # # 归一化损失函数
        # normalized_loss_psc = loss_psc / global_max_loss_psc
        # normalized_loss_criterion = loss_criterion / global_max_loss_criterion

        # 动态调整权重
        weight_criterion = (epoch / num_epochs)
        weight_psc = 1 - weight_criterion

        # 合并损失
        # loss = weight_criterion * normalized_loss_criterion + weight_psc * normalized_loss_psc
        # loss = weight_criterion * loss_criterion + weight_psc * loss_psc*weight_factor
        loss = weight_criterion * loss_criterion + weight_psc * loss_psc * weight_psc_
        # loss = loss_criterion
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
torch.save(model.state_dict(), 'weights_wc.pth')

model.load_state_dict(torch.load('weights_wc.pth'))


test_files = [
    '..\\features\\biggan_test_features.pth',
    '..\\features\\wukong_test_features.pth',
    '..\\features\\glide_test_features.pth',
    '..\\features\\sdv4_test_features.pth',
    '..\\features\\sdv5_test_features.pth',
    '..\\features\\VQDM_test_features.pth',
    '..\\features\\ilsvrc_features.pth',
    '..\\features\\ilsvrc_adm_features.pth',
    '..\\features\\SAN_test_features.pth'
]
test_labels = [1, 1, 1, 1, 1, 1, 0, 1, 1]
X_tests, y_tests = prepare_test_data(test_files, test_labels)
print(len(X_tests), len(y_tests))

# clf = LogisticRegression(max_iter=1000)
# clf.fit(X_train, y_train)

# 定义一个函数来计算和打印准确率
def evaluate_model(clf, X_test, y_test, dataset_name):
    # Assuming X_test and y_test are tensors, first ensure they are detached and converted
    z, outputs = clf(X_test)
    _, y_pred = torch.max(outputs, 1)  # Get the predicted classes

    # Detach and convert to numpy for sklearn functions
    y_pred = y_pred.detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()

    accuracy = accuracy_score(y_test_np, y_pred)
    accuracy_scores.append(accuracy)
    print(f"Linear Classifier Accuracy({dataset_name}): {accuracy:.4f}")

accuracy_scores = []
# 评估每个测试集的准确率
datasets = ['biggan', 'wukong', 'glide', 'sdv4', 'sdv5', 'VQDM', 'ilsvrc', 'ilsvrc_adm']
for i, dataset_name in enumerate(datasets):
    X_test = X_tests[i].to(device)
    # print(X_test.shape)
    y_test = y_tests[i].to(device)
    evaluate_model(model, X_test, y_test, dataset_name)

# 将准确率保存到文本文件中
with open('26-0.5-1-5.txt', 'w') as file:
    for name, score in zip(datasets, accuracy_scores):
        file.write(f"{name}: {score:.4f}\n")

# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 使用 Seaborn 设置美观的绘图风格
# sns.set(style="whitegrid")
#
# # 确保中文显示正常
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 或使用其他支持中文的字体
#
# # 创建条形图
# plt.figure(figsize=(10, 8))
# bars = plt.bar(datasets, accuracy_scores, color=sns.color_palette("viridis", len(datasets)), edgecolor='black')
#
# # 添加纹理
# for bar in bars:
#     bar.set_hatch('/')  # 这会给条形图添加斜线填充
#
# plt.xlabel('dataset', fontsize=14)
# plt.ylabel('accuracy', fontsize=14)
# plt.title('contrastive classifier', fontsize=16)
# plt.xticks(rotation=45, fontsize=12)  # 将x轴标签旋转45度以便阅读
# plt.ylim([0, 1])  # 设置y轴的范围
# plt.tight_layout()  # 自动调整布局
#
# # 添加一个图例
# plt.legend(['accuracy'], loc='upper left')
#
# # 保存和显示图形
# plt.savefig('k-means_accuracy_bar_chart.png')
# plt.show()
