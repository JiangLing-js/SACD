from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn.functional as F
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
def load_and_pool_features(file_path):
    features = torch.load(file_path)
    pooled_features = {}
    for filename, embedding in features.items():
        # 对嵌入进行最大池化操作，得到[256]的向量
        max_pooled_embedding = F.adaptive_max_pool2d(embedding, (1, 1)).view(-1)
        pooled_features[filename] = max_pooled_embedding
    return pooled_features

def prepare_data(features, label):
    X = []
    y = []
    for embedding in features.values():
        X.append(embedding.numpy())
        y.append(label)
    return X, y

def prepare_and_convert_data(file_paths, labels):
    X, y = [], []
    for file_path, label in zip(file_paths, labels):
        features = load_and_pool_features(file_path)
        X_batch, y_batch = prepare_data(features, label)
        X.extend(X_batch)
        y.extend(y_batch)
    return torch.tensor(X).numpy(), torch.tensor(y).numpy()

def prepare_test_data(file_paths, labels):
    X, y = [], []
    for file_path, label in zip(file_paths, labels):
        features = load_and_pool_features(file_path)
        X_batch, y_batch = prepare_data(features, label)
        X.append(torch.tensor(X_batch).numpy())
        y.append(torch.tensor(y_batch).numpy())
    return X, y
class KMeansClassifier:
    def __init__(self, n_clusters=2, n_init=10):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
        self.centers = []

    def fit(self, X, y):
        # 为每个类别训练一个K-means聚类
        self.labels = np.unique(y)
        self.centers = []
        for label in self.labels:
            cluster_data = X[y == label]
            if len(cluster_data) > 1:  # 确保类别内有足够的数据进行聚类
                self.kmeans.fit(cluster_data)
                self.centers.append((label, self.kmeans.cluster_centers_))

    def predict(self, X):
        predictions = []
        for x in X:
            # 计算x到所有中心的距离，并找到最近的中心
            min_dist = np.inf
            pred_label = None
            for label, centers in self.centers:
                distances = np.linalg.norm(x - centers, axis=1)
                min_center_dist = np.min(distances)
                if min_center_dist < min_dist:
                    min_dist = min_center_dist
                    pred_label = label
            predictions.append(pred_label)
        return np.array(predictions)

# 特征文件路径和对应标签
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
    ('..\\features\\SAN_test_features.pth', 1)
]

# 准备训练数据和测试数据
train_files = ['..\\features\\ai_test_features.pth', '..\\features\\nature_test_features.pth']
train_labels = [1, 0]
X_train, y_train = prepare_and_convert_data(train_files, train_labels)
print(X_train.shape, y_train.shape)

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

# 使用KMeansClassifier
kmeans_clf = KMeansClassifier(n_clusters=2)
kmeans_clf.fit(X_train, y_train)

accuracy_scores = []
# 定义一个新的评估函数
def evaluate_kmeans_model(clf, X_test, y_test, dataset_name):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print(f"KMeans Classifier Accuracy({dataset_name}): {accuracy:.4f}")

datasets = ['biggan', 'wukong', 'glide', 'sdv4', 'sdv5', 'VQDM', 'ilsvrc', 'ilsvrc_adm']
# 评估每个测试集的准确率
for i, dataset_name in enumerate(datasets):
    X_test = np.array(X_tests[i])  # 确保输入是NumPy数组
    y_test = np.array(y_tests[i])
    evaluate_kmeans_model(kmeans_clf, X_test, y_test, dataset_name)

# 将准确率保存到文本文件中
with open('k-means.txt', 'w') as file:
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
# plt.title('k-means model', fontsize=16)
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
