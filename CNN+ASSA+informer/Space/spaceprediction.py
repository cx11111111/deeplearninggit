import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei以支持中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号乱码问题

# 读取数据
turbines_location = pd.read_csv(r'C:\Users\22279\deeplearninggit\CNN+ASSA+informer\data\turb_location.CSV')
plt.figure()
plt.scatter(turbines_location['x'], turbines_location['y'], marker='o')  # s是点的大小
for index, row in turbines_location.iterrows():
    plt.text(row['x']+100, row['y'], str(row['TurbID']), fontsize=9)  # 将编号放在点的右侧
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Wind Turbine Locations')
plt.show()

# 确定网格尺寸
grid_size_x=6
grid_size_y=26
x_max,y_max=turbines_location['x'].max(),turbines_location['y'].max()
x_min,y_min=turbines_location['x'].min(),turbines_location['y'].min()

# 映射函数：将坐标转换为网格索引
def map_to_grid(x, y, x_min, x_max, y_min, y_max, grid_size_x, grid_size_y):
    x_index = np.floor((x - x_min) / (x_max - x_min) * grid_size_x)
    y_index = np.floor((y - y_min) / (y_max - y_min) * grid_size_y)
    return int(x_index), int(y_index)
# 更新风机位置到网格坐标
turbines_location['grid_x'], turbines_location['grid_y'] = zip(*turbines_location.apply(lambda row: map_to_grid(row['x'], row['y'], x_min, x_max, y_min, y_max, grid_size_x, grid_size_y), axis=1))
#turbines_location.to_csv('坐标.csv', index=False, encoding='utf-8')
plt.figure()
plt.scatter(turbines_location['grid_x'], turbines_location['grid_y'], marker='o')  # s是点的大小
for index, row in turbines_location.iterrows():
    plt.text(row['grid_x']+0.1, row['grid_y'], str(row['TurbID']), fontsize=9)  # 将编号放在点的右侧
plt.xlabel('Grid X')
plt.ylabel('Grid Y')
plt.title('Wind Turbine Locations on Grid')
plt.show()

turbines_power = pd.read_csv(r'C:\Users\22279\deeplearninggit\CNN+ASSA+informer\data\all_turb.csv')
turbines_power=turbines_power[['TurbID','Day','Tmstamp','Patv']]
base_date=datetime(2022,1,1)
turbines_power['Date']=turbines_power['Day'].apply(lambda x: base_date+pd.Timedelta(days=x-1))
turbines_power['date']=turbines_power.apply(lambda row:datetime(row['Date'].year, row['Date'].month, row['Date'].day,int(row['Tmstamp'].split(':')[0]),int(row['Tmstamp'].split(':')[1])), axis=1)
columns=['date']+[col for col in turbines_power if col!='date']
turbines_power=turbines_power[columns]
turbines_power.drop(['Day','Tmstamp','Date'],axis=1,inplace=True)
turbines_power['Patv']=turbines_power['Patv'].fillna(turbines_power['Patv'].mean())
# print(turbines_power)
# 确保时间格式正确
turbines_power['date']=pd.to_datetime(turbines_power['date'])
#数据合并
turbines=turbines_power.merge(turbines_location,on='TurbID',how='left')
#turbines.to_csv('热力图数据.csv',index=False,encoding='utf-8')
print(turbines)
#turbines=pd.read_csv('热力图数据.csv')

save_folder=r'C:\Users\22279\Desktop\大论文数据\空间发电热力图'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

start_time=datetime.strptime(str(turbines['date'].min()), '%Y-%m-%d %H:%M:%S')
end_time=datetime.strptime(str(turbines['date'].max()), '%Y-%m-%d %H:%M:%S')
time_step=timedelta(minutes=10)

current_time=start_time
vmin=turbines['Patv'].min()
vmax=turbines['Patv'].max()

power_matrices=np.zeros((245*24*6,grid_size_y+1,grid_size_x+1))
time_idx=0
while current_time<=end_time:
    selected_data=turbines[turbines['date']==current_time]

    power_grid=np.zeros((grid_size_y+1,grid_size_x+1))
    for index,row in selected_data.iterrows():
        grid_x=int(row['grid_x'])
        grid_y=int(row['grid_y'])
        power_grid[grid_y,grid_x]=row['Patv']

    #print(power_grid)
    power_matrices[time_idx]=power_grid

    # 绘制热力图
    plt.figure()
    sns.heatmap(power_grid, annot=False, cmap='viridis', fmt=".0f",vmin=vmin,vmax=vmax,cbar_kws={'label':'有功功率'})
    #plt.title(f'Power Output Heatmap at {current_time}')
    plt.xlabel('x')
    plt.ylabel('y')
    formatted_time=current_time.strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig(os.path.join(save_folder,f'Power_{formatted_time}.png'))
    plt.close()
    current_time=current_time+time_step
    time_idx+=1

np.save(os.path.join(save_folder,'power_matrices.npy'),power_matrices)
