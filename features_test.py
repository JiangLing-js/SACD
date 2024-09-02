import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载保存的特征
features_ai = torch.load('features/image_features.pth')
features_nature = torch.load("features/nature_test.pth")

# 准备数据
X = []
y = []

for filename, embedding in features_ai.items():
    X.append(embedding.numpy())
    y.append('AI')

for filename, embedding in features_nature.items():
    X.append(embedding.numpy())
    y.append('Nature')

X = np.array(X)
y = np.array(y)

# 检查数据形状
print("Feature shape:", X.shape)
print("Labels:", y)

# 进行t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# 检查t-SNE结果形状
print("t-SNE result shape:", X_tsne.shape)

# 可视化t-SNE结果
plt.figure(figsize=(10, 8))

# 使用不同的颜色表示不同的类别
colors = {'AI': 'r', 'Nature': 'g'}

for label in np.unique(y):
    indices = y == label
    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], c=colors[label], label=label, alpha=0.6)

plt.title('t-SNE visualization of AI and Nature features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()

# 保存图像到文件
plt.savefig('tsne_visualization.png')

# 显示图像
plt.show()

print("t-SNE visualization saved successfully.")
