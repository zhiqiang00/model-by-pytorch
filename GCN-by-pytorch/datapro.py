import pandas as pd
import numpy as np

#设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# 导入数据C:\Users\zhiqiang\Desktop\coding\model-by-pytorch\GCN-by-pytorch\data\cora\cora.content
raw_data = pd.read_csv('data\cora\cora.content', sep='\t', header=None)
num = raw_data.shape[0]

# 将论文编号转为[0 2708]
a = list(raw_data.index)
b = list(raw_data[0])
c = zip(b, a)
map = dict(c)

# 提取词向量，成为特征矩阵
features = raw_data.iloc[:,1:-1]
print(features)

# 提取标签，进行独热编码
labels = pd.get_dummies(raw_data[1434])
print(labels.head(3))

# 导入论文引用数据
raw_data_cites = pd.read_csv('data/cora/cora.cites', sep='\t', header=None)
matrix = np.zeros((num, num))
for i,j in zip(raw_data_cites[0], raw_data_cites[1]):
    x = map[i]
    y = map[j]
    matrix[x][y] = matrix[y][x] = 1
print(sum(matrix))
