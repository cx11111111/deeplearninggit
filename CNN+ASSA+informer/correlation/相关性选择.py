import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

data_folder=r'C:\Users\22279\deeplearninggit\CNN+ASSA+informer\data'
save_folder=r'C:\Users\22279\Desktop\大论文数据\相关性'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for i in range(1,135):
    file_name = os.path.join(data_folder, f'turb_{i}.csv')
    df = pd.read_csv(file_name)
    TARGET_COL = 'Patv'
    df = df[df[TARGET_COL] >= 0]

    base_date=datetime(1990,1,1)
    df['Date']=df['Day'].apply(lambda x: base_date+pd.Timedelta(days=x-1))
    df['date']=df.apply(lambda row:datetime(row['Date'].year, row['Date'].month, row['Date'].day,int(row['Tmstamp'].split(':')[0]),int(row['Tmstamp'].split(':')[1])), axis=1)
    columns=[col for col in df if col!='date']
    df=df[columns]
    df.drop(['TurbID','Day','Tmstamp','Date'],axis=1,inplace=True)
    for col in ['Wspd','Wdir','Etmp','Itmp','Ndir','Pab1','Pab2','Pab3','Prtv','Patv']:
        df[col]=df[col].fillna(df[col].mean())

    correlation_matrix = df.corr()
# 将相关系数矩阵转换为对称矩阵
# 绘制相关性热力图
    custom_labels=['风速','风向','环境温度','机舱温度','机舱角','叶片1俯仰角','叶片2俯仰角','叶片3俯仰角','无功功率','有功功率']

    fig,ax=plt.subplots(figsize=(16, 16))  # 设置画面大小
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    heatmap=sns.heatmap(
        correlation_matrix,
        annot=True,
        annot_kws={"weight": "bold","fontsize":20},
        vmax=1,
        square=True,
        cmap="Blues",
        xticklabels=custom_labels,
        yticklabels=custom_labels,
        cbar=False
    )
    plt.xticks(fontsize=20,fontweight='bold',rotation=45)
    plt.yticks(fontsize=20,fontweight='bold',rotation=45)
    divider=make_axes_locatable(ax)
    cax=divider.append_axes("right", size="5%", pad=0.2)
    cbar=fig.colorbar(heatmap.collections[0], cax=cax)
    #cbar=heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    #plt.title(f'相关性热力图',fontsize=20)
    plt.savefig(os.path.join(save_folder, f'Turb_{i}.png'))
    plt.close()
