import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from contrastive_classifier import FeatureClassifierModel, psc_loss, compute_prototype

def load_and_pool_features(file_path):
    features = torch.load(file_path)
    #print(features)
    return features

def prepare_data(features, label):
    X = []
    y = []
    for embedding in features.values():
        # 确保 embedding 转换为 PyTorch 张量
        tensor_embedding = torch.tensor(embedding, dtype=torch.float32).squeeze(0)  # 转换为张量并去掉多余的维度
        X.append(tensor_embedding)  # 将张量添加到列表中
        y.append(label)
    return X, y

def prepare_and_convert_data(file_paths, labels):
    X, y = [], []
    for file_path, label in zip(file_paths, labels):
        features = load_and_pool_features(file_path)
        X_batch, y_batch = prepare_data(features, label)
        X.extend(X_batch)
        y.extend(y_batch)
    # 转换为张量
    X_tensor = torch.stack(X)  # 使用 torch.stack 将列表转换为一个批次张量
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
    return X, y

# 特征文件路径和对应标签
feature_files_labels = [
    ('..\\tf_features\\ai_test_features.pth', 1),
    ('..\\tf_features\\nature_test_features.pth', 0),
    ('..\\tf_features\\biggan_test_features.pth', 1),
    ('..\\tf_features\\wukong_test_features.pth', 1),
    ('..\\tf_features\\glide_test_features.pth', 1),
    ('..\\tf_features\\sdv4_test_features.pth', 1),
    ('..\\tf_features\\sdv5_test_features.pth', 1),
    ('..\\tf_features\\VQDM_test_features.pth', 1),
    ('..\\tf_features\\ilsvrc_features.pth', 0),
    ('..\\tf_features\\ilsvrc_adm_features.pth', 1),
    #('..\\tf_features\\SAN_test_features.pth', 1)
]

train_files = ['..\\tf_features\\ai_test_features.pth', '..\\tf_features\\nature_test_features.pth']
train_labels = [1, 0]

X_train, y_train = prepare_and_convert_data(train_files, train_labels)
print(X_train.shape, y_train.shape)

# Dataset and DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# 训练线性分类器
model = FeatureClassifierModel(num_channels=256, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epochs = 26  # 根据需要调整
tau = 1.0
model.train()
alpha = 1
global_max_loss_psc = 1.0
global_max_loss_criterion = 1.0
moving_average_decay = 0.9
weight_psc_ = 1

for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        z, outputs = model(images)
        prototypes = compute_prototype(z, labels)
        loss_psc = psc_loss(z, labels, prototypes, tau)
        loss_criterion = criterion(outputs, labels)

        weight_criterion = (epoch / num_epochs)
        weight_psc = 1 - weight_criterion

        loss = weight_criterion * loss_criterion + weight_psc * loss_psc * weight_psc_
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
torch.save(model.state_dict(), 'tf_weights.pth')

test_files = [
    '..\\tf_features\\biggan_test_features.pth',
    '..\\tf_features\\wukong_test_features.pth',
    '..\\tf_features\\glide_test_features.pth',
    '..\\tf_features\\sdv4_test_features.pth',
    '..\\tf_features\\sdv5_test_features.pth',
    '..\\tf_features\\VQDM_test_features.pth',
    '..\\tf_features\\ilsvrc_features.pth',
    '..\\tf_features\\ilsvrc_adm_features.pth',
    #'..\\tf_features\\SAN_test_features.pth'
]
test_labels = [1, 1, 1, 1, 1, 1, 0, 1]
X_tests, y_tests = prepare_test_data(test_files, test_labels)
print(len(X_tests), len(y_tests))

def evaluate_model(model, X_test, y_test, dataset_name):
    # Stack X_test if it's a list of tensors
    if isinstance(X_test, list):
        X_test = torch.stack(X_test)

    # Ensure y_test is a tensor
    if isinstance(y_test, list):
        y_test = torch.tensor(y_test, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)[1]  # Assuming the model returns (features, outputs)
        _, y_pred = torch.max(outputs, 1)  # Get the predicted classes

        # Detach and convert to numpy for sklearn functions
        y_pred_np = y_pred.cpu().numpy()
        y_test_np = y_test.cpu().numpy()

        accuracy = accuracy_score(y_test_np, y_pred_np)
        accuracy_scores.append(accuracy)
        print(f"Linear Classifier Accuracy({dataset_name}): {accuracy:.4f}")

accuracy_scores = []
datasets = ['biggan', 'wukong', 'glide', 'sdv4', 'sdv5', 'VQDM', 'ilsvrc', 'ilsvrc_adm']
for i, dataset_name in enumerate(datasets):
    X_test = X_tests[i]
    y_test = y_tests[i]
    evaluate_model(model, X_test, y_test, dataset_name)

with open('tf_results.txt', 'w') as file:
    for name, score in zip(datasets, accuracy_scores):
        file.write(f"{name}: {score:.4f}\n")
