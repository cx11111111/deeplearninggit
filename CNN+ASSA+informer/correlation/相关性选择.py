import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# 读取菜品销售量数据
cor1 = pd.read_csv('turb_1.csv')
print('相关系数矩阵为：\n', cor1.corr())
df = pd.DataFrame(cor1.corr())
# 保存到 Excel 文件
df.to_excel('correlation_matrix.xlsx', index=False)
print("相关系数矩阵已保存到 correlation_matrix.xlsx 文件中。")
cor2=pd.read_excel('correlation_matrix.xlsx')
correlation_matrix = cor2.corr()
# 将相关系数矩阵转换为对称矩阵
# 绘制相关性热力图
plt.subplots(figsize=(16, 16))  # 设置画面大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.heatmap(correlation_matrix, annot=True, annot_kws={"weight": "bold"}, vmax=1, square=True, cmap="Blues", xticklabels=cor2.columns, yticklabels=cor2.columns)
plt.show()