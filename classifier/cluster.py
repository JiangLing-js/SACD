import torch
import torch.nn.functional as F
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_and_pool_features(file_path):
    features = torch.load(file_path)
    pooled_features = {}
    for filename, embedding in features.items():
        # 对嵌入进行最大池化操作，得到[256]的向量
        max_pooled_embedding = F.adaptive_max_pool2d(embedding, (1, 1)).view(-1)
        pooled_features[filename] = max_pooled_embedding
    return pooled_features

# 加载并处理特征文件
ai_features = load_and_pool_features('..\\features\\ai_test_features.pth')
nature_features = load_and_pool_features('..\\features\\nature_test_features.pth')
biggan_features = load_and_pool_features('..\\features\\biggan_test_features.pth')
wukong_features = load_and_pool_features('..\\features\\wukong_test_features.pth')
SAN_features = load_and_pool_features('..\\features\\SAN_test_features.pth')

# 将特征和标签准备为数据
def prepare_data(features, label):
    X = []
    y = []
    for embedding in features.values():
        X.append(embedding.numpy())
        y.append(label)
    return X, y

X_ai, y_ai = prepare_data(ai_features, 0)
X_nature, y_nature = prepare_data(nature_features, 1)
X_biggan, y_biggan = prepare_data(biggan_features, 2)
X_wukong, y_wukong = prepare_data(wukong_features, 3)
#X_SAN, y_SAN = prepare_data(SAN_features, 4)

# 合并所有数据
X = X_ai + X_nature + X_biggan + X_wukong
y = y_ai + y_nature + y_biggan + y_wukong

# 转换为numpy数组
X = torch.tensor(X).numpy()
y = torch.tensor(y).numpy()

# 聚类
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# 预测
y_kmeans = kmeans.predict(X)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# 可视化 t-SNE 结果
plt.figure(figsize=(12, 8))
colors = ['r', 'g', 'b', 'y']  # 定义四种不同的颜色
labels = ['ADM', 'real', 'biggan', 'wukong']  # 数据集标签

# 根据标签值设置颜色和标签
for i in range(4):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], c=colors[i], label=labels[i], alpha=0.6)

plt.title('t-SNE Visualization of Different Features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
# 保存图像到文件
plt.savefig('TSNE_mymodel.png')
plt.show()
