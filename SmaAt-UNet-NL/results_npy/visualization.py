import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
# 加载数据
preds = np.load('Unet/results_npy/preds_denorm.npy')  # shape: (B, T, 27, 7)
trues = np.load('Unet/results_npy/trues_denorm.npy')

# 设置保存路径
save_dir = 'Unet'
os.makedirs(save_dir, exist_ok=True)

# 设置可视化样本数量
num_samples_to_plot = 294
frame_idx = 12  # 可选时间帧

for i in range(num_samples_to_plot):
    for j in range(frame_idx):
        pred = preds[i, j]   # shape: (27, 7)
        true = trues[i, j]   # shape: (27, 7)
        diff = np.abs(pred - true)

        fig, axs = plt.subplots(1, 3, figsize=(12,9))
        im0 = axs[0].imshow(true, cmap='viridis',vmin=0, vmax=1600)
        #axs[0].set_title("真实功率分布",fontsize=15)
        axs[0].set_xticks(np.arange(7))
        axs[0].set_yticks(np.arange(27))
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        #axs[0].set_aspect('equal')
        cbar0=plt.colorbar(im0, ax=axs[0])
        cbar0.ax.tick_params(labelsize=15)


        im1 = axs[1].imshow(pred, cmap='viridis',vmin=0, vmax=1600)
        #axs[1].set_title("预测功率分布",fontsize=15)
        axs[0].set_xticks(np.arange(7))
        axs[0].set_yticks(np.arange(27))
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        #axs[0].set_aspect('equal')
        cbar1=plt.colorbar(im1, ax=axs[1])
        cbar1.ax.tick_params(labelsize=15)

        im2 = axs[2].imshow(diff, cmap='Reds',vmin=0, vmax=1600)
        #axs[2].set_title("绝对误差图",fontsize=15)
        axs[0].set_xticks(np.arange(7))
        axs[0].set_yticks(np.arange(27))
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        #axs[0].set_aspect('equal')
        cbar2=plt.colorbar(im2, ax=axs[2])
        cbar2.ax.tick_params(labelsize=15)



        for ax in axs:
            ax.axis("off")

        plt.suptitle(f"样本 {i} - 第 {j} 帧预测对比", fontsize=14)
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{i}_frame_{j}_heatmap.png"))
        plt.close()

print(f"✅ 已保存热力图至 {save_dir}/")
