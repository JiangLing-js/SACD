import numpy as np

def read_accuracy_file(filepath):
    accuracy_dict = {}
    with open(filepath, 'r') as file:
        for line in file:
            name, score = line.strip().split(': ')
            accuracy_dict[name] = float(score)
    return accuracy_dict

# 读取各个文件的数据
contrastive_acc = read_accuracy_file('26-1-0.02-1.txt')
kmeans_acc = read_accuracy_file('k-means.txt')
linear_acc = read_accuracy_file('linear.txt')

# 确保数据集名称相同且顺序一致
datasets = list(contrastive_acc.keys())
contrastive_scores = [contrastive_acc[name] for name in datasets]
kmeans_scores = [kmeans_acc[name] for name in datasets]
linear_scores = [linear_acc[name] for name in datasets]

import matplotlib.pyplot as plt

# 设置条形宽度
bar_width = 0.25
index = np.arange(len(datasets))

# 创建条形图
fig, ax = plt.subplots(figsize=(12, 8))
bar1 = ax.bar(index, contrastive_scores, bar_width, label='Contrastive Classifier')
bar2 = ax.bar(index + bar_width, kmeans_scores, bar_width, label='K-Means Classifier')
bar3 = ax.bar(index + 2 * bar_width, linear_scores, bar_width, label='Linear Classifier')

# 添加图表标题和标签
ax.set_xlabel('Datasets', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy Comparison of Three Models on Different Datasets', fontsize=14)
ax.set_xticks(index + bar_width)
ax.set_xticklabels(datasets, rotation=45)
ax.set_ylim([0.90, 1])  # Assuming accuracy is between 0.90 and 1 for better visibility

# 添加图例
ax.legend()

# 显示图形
plt.tight_layout()
plt.show()


