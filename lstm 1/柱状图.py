import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 读取 CSV 文件
file_path = r'C:\Users\22279\Desktop\风机1指标.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 设置绘图参数
models = data['Model']  # 模型名称
metrics = data.columns[1:]  # 指标列名：MSE、RMSE、MAE
bar_width = 0.25  # 每个柱的宽度
x = np.arange(len(metrics))  # 横坐标位置

hatch_styles=['///','\\\\\\','xxx']

# 绘制柱状图
plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.bar(x + i * bar_width, data.iloc[i, 1:], bar_width, label=model,
            color='white', edgecolor='black',hatch=hatch_styles[i])

# 设置横纵坐标标签和标题
plt.xlabel('误差指标', fontsize=14)
plt.ylabel('误差值', fontsize=14)
plt.title('不同模型的误差对比', fontsize=16)

# 设置横坐标刻度
plt.xticks(x + bar_width, metrics, fontsize=12)

# 添加图例
plt.legend(fontsize=14)


# 显示图表
plt.tight_layout()
plt.show()

