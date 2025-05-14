import pandas as pd
import os
import re

# 设置你的文件夹路径
folder_path = 'C:/Users/22279/Desktop/大论文数据/CNNASSA-Informer/'  # 替换成你自己的路径

# 获取所有以“指标turb_”开头的 csv 文件
file_list = [f for f in os.listdir(folder_path) if f.startswith('指标turb_') and f.endswith('.csv')]

# 提取编号并排序
def extract_id(filename):
    match = re.search(r'指标turb_(\d+)\.csv', filename)
    return int(match.group(1)) if match else -1

# 排序文件列表
file_list_sorted = sorted(file_list, key=extract_id)

all_data = []

for file_name in file_list_sorted:
    turbine_id = extract_id(file_name)
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    df['风机编号'] = turbine_id
    all_data.append(df)

# 合并所有风机数据
combined_df = pd.concat(all_data, ignore_index=True)

# 保存结果
combined_df.to_csv('C:/Users/22279/Desktop/大论文数据/CNNASSA-Informer/风机指标汇总.csv', index=False, encoding='utf-8-sig')


# import matplotlib.pyplot as plt
# import pandas as pd
# from matplotlib import rcParams
#
# data=pd.read_csv(r'C:\Users\22279\Desktop\大论文数据\风机1预测.csv')
#
# rcParams['font.sans-serif'] = ['SimHei']
# rcParams['axes.unicode_minus'] = False
#
# plt.figure()
# plt.plot(range(200),data['real'][400:600], color='red', label="True values", alpha=0.6)
# plt.plot(range(200),data['GRU predict'][400:600], label="GRU", alpha=0.6)
# plt.plot(range(200),data['CNN-ASSA-Informer'][400:600], label="CNN-ASSA-Informer", alpha=0.6)
# plt.plot(range(200),data['LSTM predict'][400:600], label="LSTM", alpha=0.6)
# plt.legend()
# plt.show()
#
#
#
#
# plt.figure()
# # 实线，较粗，无标记
# plt.plot(range(200), data['real'][400:600],  marker='*', markersize=6,markevery=5,label="真实值", alpha=0.6)
# # 虚线，带圆形标记，每隔10个点标记一次
# plt.plot(range(200), data['GRU predict'][400:600],  marker='o', markersize=6, markevery=10, label="Informer预测值", alpha=0.6)
# # 点划线，带三角形标记，每隔15个点标记一次
# plt.plot(range(200), data['CNN-ASSA-Informer'][400:600],  marker='^', markersize=6, markevery=15, label="CNN-ASSA-Informer预测值", alpha=0.6)
# # 点线，带方形标记，每隔20个点标记一次
# plt.plot(range(200), data['LSTM predict'][400:600],  marker='s', markersize=6, markevery=20, label="LSTM预测值", alpha=0.6)
#
# plt.xlabel("时间（10分钟）",fontsize=10)
# plt.ylabel("功率(千瓦）",fontsize=11)
# plt.legend()
# plt.show()


