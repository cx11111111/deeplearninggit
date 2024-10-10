import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from vmdpy import VMD
# 读取CSV文件
file_path = '2020.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)
# 提取fuhe列
aqi = df['Target'].values
# VMD参数设置
# 加载优化的参数
pop = loadmat('RIME_para.mat')['out_xposbest'].reshape(-1,)
BEST_alpha = int(pop[0])  # 惩罚稀疏
BEST_K = int(pop[1])      # 最佳K

print('最佳保真项惩罚系数为：', BEST_alpha)
print('最佳模态数为：', BEST_K)

# VMD参数设置
alpha = BEST_alpha      # 保真项惩罚系数
tau = 0                 # 噪声容忍度
K = BEST_K              # 模态数
DC = False              # 是否包括DC成分
init = 1                # 初始化模式
tol = 1e-7              # 收敛容差
# 进行VMD分解
imfs, u_hat, omega = VMD(aqi, alpha, tau, K, DC, init, tol)
# 绘制IMF分解图
plt.figure(figsize=(12, 9))
num_imfs = imfs.shape[0]
for i in range(num_imfs):
    plt.subplot(num_imfs, 1, i + 1)
    plt.plot(imfs[i, :])
    plt.title(f'IMF {i + 1}')
plt.tight_layout()
plt.show()
# 保存IMF结果到CSV文件
imf_df = pd.DataFrame(imfs.transpose(), columns=[f'IMF{i + 1}' for i in range(num_imfs)])
output_file_path = '2020_imf_results.csv'
imf_df.to_csv(output_file_path, index=False)

print(f'IMF results saved to {output_file_path}')

# 创建IMF的DataFrame
imf_df = pd.DataFrame(imfs.transpose(), columns=[f'IMF{i + 1}' for i in range(num_imfs)])

# 将原始DataFrame拆分为前半部分和后半部分
df_part1 = df.iloc[:, :-1]  # 除倒数两列的所有列
df_part2 = df.iloc[:, -1:]  # 最后两列

# 将IMF结果插入到原始DataFrame的倒数第二列之后
combined_df = pd.concat([df_part1, imf_df, df_part2], axis=1)

# 保存拼接后的DataFrame为CSV文件
output_file_path = '../data/2020_combined_results.csv'
combined_df.to_csv(output_file_path, index=False)

print(f'Combined results saved to {output_file_path}')

