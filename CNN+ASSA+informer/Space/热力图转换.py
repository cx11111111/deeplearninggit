import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei以支持中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号乱码问题

wind_power=pd.read_csv(r'C:\Users\22279\deeplearninggit\CNN+ASSA+informer\data\all_turb.csv')
#wind_power=wind_power[['TurbID','Patv']]
selected_turbines=wind_power['TurbID'].unique()[67:]
subset=wind_power[wind_power['TurbID'].isin(selected_turbines)]

# 绘制多台风机功率的箱线图
plt.figure(figsize=(12, 6))
sns.boxplot(x='TurbID', y='Patv', data=subset)
plt.xlabel('风机',fontsize=15)
plt.ylabel('有功功率（kW）',fontsize=15)
plt.tick_params(direction='in',length=2)
plt.tight_layout()
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('箱线图2.png')


# base_date=datetime(2022,1,1)
# wind_power['Date']=wind_power['Day'].apply(lambda x: base_date+pd.Timedelta(days=x-1))
# wind_power['timestamp']=wind_power.apply(lambda row:datetime(row['Date'].year, row['Date'].month, row['Date'].day,int(row['Tmstamp'].split(':')[0]),int(row['Tmstamp'].split(':')[1])), axis=1)
#
#
# wind_power.set_index('timestamp',inplace=True)
# # 按时间戳分组，计算所有风机的发电功率均值和方差
# grouped = wind_power.groupby('timestamp')['Patv'].agg(['mean', 'var']).reset_index()
# grouped=grouped.iloc[::24,:]
# #grouped = wind_power.resample('2H')['Patv'].agg(['mean', 'var']).reset_index()
#
# # 绘制均值和方差随时间变化的图（可以在同一图中展示）
# plt.figure(figsize=(14, 7))
# plt.plot( grouped['mean'], label='均值')
# plt.title("风电场功率均值随时间变化")
# plt.xlabel("时间")
# plt.ylabel("有功功率（kW）")
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
#
# plt.figure(figsize=(14, 7))
# plt.plot(grouped['var'], label='方差')
# plt.title("风电场功率方差随时间变化")
# plt.xlabel("时间")
# plt.ylabel("有功功率（kW）")
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


