import os
import numpy as np
import matplotlib.pyplot as plt

# 文件路径和参数设置
base_path = ''  # 替换为你的文件路径
experiments = [
    {'folder': '26-0.5-1',
     'files': ['26-0.5-1-1.txt', '26-0.5-1-2.txt', '26-0.5-1-3.txt', '26-0.5-1-4.txt', '26-0.5-1-5.txt']},
    {'folder': '26-1-1', 'files': ['26-1-1-1.txt', '26-1-1-2.txt', '26-1-1-3.txt', '26-1-1-4.txt', '26-1-1-5.txt']},
    {'folder': '26-1-0.1',
     'files': ['26-1-0.1-1.txt', '26-1-0.1-2.txt', '26-1-0.1-3.txt', '26-1-0.1-4.txt', '26-1-0.1-5.txt']},
    {'folder': '26-1-0.02',
     'files': ['26-1-0.02-1.txt', '26-1-0.02-2.txt', '26-1-0.02-3.txt', '26-1-0.02-4.txt', '26-1-0.02-5.txt']}
]


def read_accuracy(file_path):
    accuracies = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                try:
                    accuracy = float(parts[1])
                    accuracies.append(accuracy)
                except ValueError:
                    continue
    return accuracies


# 存储所有实验结果
results = {}

for exp in experiments:
    accuracies = []
    for file_name in exp['files']:
        file_path = os.path.join(base_path, exp['folder'], file_name)
        accuracies.extend(read_accuracy(file_path))
    results[exp['folder']] = accuracies

# 画图
fig, ax = plt.subplots()

# 定义颜色
colors = ['#EAEAEA', '#91BFFA', '#FF7F00', '#DDE0C1']

folders = {'26-0.5-1': "tau:0.5, w_psc:1.0", '26-1-1': "tau:1, w_psc:1.0",
           '26-1-0.1': "tau:1, w_psc:0.1", '26-1-0.02': "tau:1, w_psc:0.02"}

# 计算每个实验的平均准确率和上下界
for i, (folder, accuracies) in enumerate(results.items()):
    bp = ax.boxplot(accuracies, positions=[i], widths=0.6, patch_artist=True)

    # 设置箱线图颜色
    for box in bp['boxes']:
        box.set(color='black', linewidth=1.5)
        box.set(facecolor=colors[i % len(colors)])
    for median in bp['medians']:
        median.set(color='black', linewidth=2)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1.5)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1.5)
    for flier in bp['fliers']:
        flier.set(marker='o', color=colors[i % len(colors)], alpha=0.5)

ax.set_xticks(range(len(results)))
ax.set_xticklabels([folders[folder] for folder in results.keys()], rotation=5, fontsize=10)
ax.set_title('Experiment Results with Custom Colors')
ax.set_ylabel('Accuracy')
plt.savefig("parameters.png")
plt.show()
