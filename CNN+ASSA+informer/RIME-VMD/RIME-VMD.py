# In[ ]
from scipy.fftpack import hilbert
from scipy.io import savemat
from vmdpy import VMD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

save_folder=r'C:\Users\22279\Desktop\大论文数据\频率分解'
data_folder=r'C:\Users\22279\deeplearninggit\CNN+ASSA+informer\data'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)


# 设置绘图字体和解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
for p in range(121,135):
    file_name = os.path.join(data_folder, f'turb_{p}.csv')
    data=pd.read_csv(file_name)
    base_date=datetime(1990,1,1)
    data['Date']=data['Day'].apply(lambda x: base_date+pd.Timedelta(days=x-1))
    data['date']=data.apply(lambda row:datetime(row['Date'].year, row['Date'].month, row['Date'].day,int(row['Tmstamp'].split(':')[0]),int(row['Tmstamp'].split(':')[1])), axis=1)
    columns=['date']+[col for col in data if col!='date']
    data=data[columns]
    data.drop(['TurbID','Day','Tmstamp','Date'],axis=1,inplace=True)
    for col in ['Wspd','Wdir','Etmp','Itmp','Ndir','Pab1','Pab2','Pab3','Prtv','Patv']:
        data[col]=data[col].fillna(data[col].mean())
    signal = data['Patv'].to_numpy()

    tau = 0
    DC = 0
    init = 1
    tol = 1e-7 #1e-6
    # In[ ]
    def vmd_decompose(series, alpha, tau,K, DC, init, tol):
        imfs_vmd, imfs_hat, omega = VMD(series, alpha, tau, K, DC, init, tol)
        df_vmd = pd.DataFrame(imfs_vmd.T)
        df_vmd.columns = ['imf' + str(i) for i in range(K)]
        return df_vmd
    def training(X):
        if len(X) == 2:
            K = int(X[1])
            alpha = X[0]
            u, u_hat, omega = VMD(data["Patv"], alpha, tau, K, DC, init, tol)
    #信息熵
            # EP = []
            # for i in range(K):
            #     H = np.abs(hilbert(u[i, :]))
            #     e1 = []
            #     for j in range(len(H)):
            #         p = H[j] / np.sum(H)
            #         e = -p * log(p, 2)
            #         e1.append(e)
            #     E = np.sum(e1)
            #     EP.append(E)
            # s = np.sum(EP) / K
    #样本熵
            EP = []
            m = 2  # 重叠长度
            for i in range(K):
                H = np.abs(hilbert(u[i, :]))
                e1 = []
                for j in range(len(H) - m):
                    s = np.sum(H[j:j + m])
                    R = np.sum(np.abs(H[j + m:j + (2 * m)] - H[j:j + m]))
                    S = np.log(R / s)
                    e1.append(S)
                E = np.mean(e1)
                EP.append(E)
            s = np.sum(EP) / K

            return s
        else:
            return 1
    def RIME(N, Max_iter, lb, ub, dim, fobj):
        # initialize position
        Best_rime = np.zeros(dim)
        Best_rime_rate = np.inf  # change this to -np.inf for maximization problems

        def Initialization(n, nd, ub, lb):

            x = np.random.rand(n, nd)
            for k in range(nd):
                for i in range(n):
                    x[i, k] = lb[k] + x[i, k] * (ub[k] - lb[k])
            return x

        Rimepop = Initialization(N, dim, ub, lb)  # Initialize the set of random solutions
        Lb = lb * np.ones(dim)  # lower boundary
        Ub = ub * np.ones(dim)  # upper boundary
        it = 1  # Number of iterations
        Convergence_curve = np.zeros(Max_iter)
        Rime_rates = np.zeros(N)  # Initialize the fitness value
        newRime_rates = np.zeros(N)
        W = 5  # Soft-rime parameters, discussed in subsection 4.3.1 of the paper

        # Calculate the fitness value of the initial position
        for i in range(N):
            Rime_rates[i] = fobj(Rimepop[i])  # Calculate the fitness value for each search agent
            # Make greedy selections
            if Rime_rates[i] < Best_rime_rate:
                Best_rime_rate = Rime_rates[i]
                Best_rime = Rimepop[i]

        # Main loop
        while it <= Max_iter:
            print('第' + str(it) + '次迭代')
            RimeFactor = (np.random.rand() - 0.5) * 2 * np.cos((np.pi * it / (Max_iter / 10))) * (
                        1 - round(it * W / Max_iter) / W)  # Parameters of Eq.(3),(4),(5)
            E = (it / Max_iter) ** 0.5  # Eq.(6)
            newRimepop = Rimepop.copy()  # Recording new populations
            normalized_rime_rates = Rime_rates / np.linalg.norm(Rime_rates)  # Parameters of Eq.(7)

            for i in range(N):
                for j in range(dim):
                    # Soft-rime search strategy
                    r1 = np.random.rand()
                    if r1 < E:
                        newRimepop[i, j] = Best_rime[j] + RimeFactor * (
                                    (Ub[j] - Lb[j]) * np.random.rand() + Lb[j])  # Eq.(3)
                    # Hard-rime puncture mechanism
                    r2 = np.random.rand()
                    if r2 < normalized_rime_rates[i]:
                        newRimepop[i, j] = Best_rime[j]  # Eq.(7)

            for i in range(N):
                # Boundary absorption
                Flag4ub = newRimepop[i, :] > ub
                Flag4lb = newRimepop[i, :] < lb
                newRimepop[i, :] = newRimepop[i, :] * (~(Flag4ub + Flag4lb)) + ub * Flag4ub + lb * Flag4lb
                newRime_rates[i] = fobj(newRimepop[i])
                # Positive greedy selection mechanism
                if newRime_rates[i] < Rime_rates[i]:
                    Rime_rates[i] = newRime_rates[i]
                    Rimepop[i, :] = newRimepop[i, :]
                    if newRime_rates[i] < Best_rime_rate:
                        Best_rime_rate = newRime_rates[i]
                        Best_rime = Rimepop[i, :]

            Convergence_curve[it - 1] = Best_rime_rate
            it = it + 1

        return Best_rime,  Best_rime_rate, Convergence_curve

    # In[ ]
    dim=2#维度
    #调节LP,UP,Max_iter，N改善模型的效果
    UP = [100000,12]#上界
    LP = [100, 6]#下届
    Max_iter = 30#最大迭代次数
    N=10#种群数
    out_xposbest, out_fvalbest, out_best_fval = RIME(N, Max_iter, LP, UP, dim, training)
    print('参数优化完成，最佳参数为：',out_xposbest)
    savemat('RIME_para.mat', {'out_xposbest': out_xposbest, 'out_fvalbest': out_fvalbest, 'out_best_fval': out_best_fval})
    # In[12]:
    # 绘制适应度曲线图，标签命名自行更改
    # plt.figure(figsize=(8, 5))
    plt.plot(out_best_fval)
    plt.title('收敛曲线')
    plt.ylabel('fitnessCurve')
    plt.xlabel('Number of iterations')
    plt.savefig(os.path.join(save_folder, f'{p}_1.png'))
    plt.close()

    # In[12]:
    K = int(out_xposbest[1])
    alpha =int( out_xposbest[0] )

    df_vmd = vmd_decompose(data["Patv"],alpha, tau,K, DC, init, tol)
    print('VMD分解完成')

    df_vmd.columns=[f'imf_{i+1}' for i in range(K)]
    df_vmd.to_csv(f'all_imfs_{p}.csv',index=False)

    # 创建IMF的DataFrame
    #imf_df = pd.DataFrame(df_vmd, columns=[f'IMF{i + 1}' for i in range(K)])
    date=data.iloc[:,0]
    combined_df1 = pd.concat([date, df_vmd], axis=1)
    output_file_path = '../data/combined_results1.csv'
    combined_df1.to_csv(output_file_path, index=False)

    # 将原始DataFrame拆分为前半部分和后半部分
    df_part1 = data.iloc[:, :-1]  # 除倒数两列的所有列
    df_part2 = data.iloc[:, -1:]  # 最后两列

    # 将IMF结果插入到原始DataFrame的倒数第二列之后
    combined_df2 = pd.concat([df_part1, df_vmd, df_part2], axis=1)

    # 保存拼接后的DataFrame为CSV文件
    output_file_path = '../data/combined_results2.csv'
    combined_df2.to_csv(output_file_path, index=False)

    # In[ ]imf图
    fig, axes = plt.subplots(nrows=len(df_vmd.columns), ncols=1)
    color_cycle = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'] * len(df_vmd.columns))

    for i, col in enumerate(df_vmd.columns):
       df_vmd[col].plot(ax=axes[i], color=next(color_cycle))
       axes[i].legend(loc="upper right")
       axes[i].tick_params(axis='y', labelsize=10)
    plt.suptitle('VMD 分解结果',fontsize=12,y=0.93)
    plt.savefig(os.path.join(save_folder, f'{p}_2.png'))
    plt.close()
    # In[ ]每个 IMF 的频率图
    fig, axes = plt.subplots(nrows=len(df_vmd.columns), ncols=1)
    color_cycle = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'] * len(df_vmd.columns))

    for i, col in enumerate(df_vmd.columns):
        # 计算每个 IMF 的频率
        imf = df_vmd[col].values
        imf_fft = np.fft.fft(imf)
        freq = np.fft.fftfreq(len(imf))

        # 绘制频率图
        ax = axes[i]
        ax.plot(freq, np.abs(imf_fft), color=next(color_cycle))
        ax.set_xlim(-0.01, 0.5)  # 限制 x 轴范围在 [0, 0.5]
        ax.tick_params(axis='y', labelsize=10)

        # 只在最后一个图显示 x 轴刻度
        if i == len(df_vmd.columns) - 1:
            ax.tick_params(axis='x', labelsize=10)
        else:
            ax.set_xticklabels([])

    plt.suptitle('频率图', fontsize=12, y=0.93)
    plt.savefig(os.path.join(save_folder, f'{p}_3.png'))
    plt.close()
