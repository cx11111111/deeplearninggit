import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
def calculate_loss(pred, true, clip_range = [0, 1]) :
    """
    同时计算 MAE, MSE 和 SSIM 损失值。
    参数:
    pred -- 预测值 (B, C, H, W) 或 (B, T, C, H, W)
    true -- 真实值 (B, C, H, W) 或 (B, T, C, H, W)
    返回:
    MAE损失值, MSE损失值, SSIM值
    """
    # 计算 MAE 和 MSE（不变部分）
    pred = np.maximum(pred, clip_range[0])
    pred = np.minimum(pred, clip_range[1])
    mae_loss = np.mean(np.abs(pred - true), axis=(0,1,2,3)).sum()  # 对所有维度计算
    mse_loss = np.mean((pred - true) ** 2, axis=(0,1,2,3)).sum()  # 对所有维度计算
    rmse_loss = np.sqrt(mse_loss)

    B,T,H,W=pred.shape
    ssim_total=0
    total_corr = 0
    for b in range(B):
        for t in range(T):
            ssim_total+=ssim(pred[b][t], true[b][t],data_range=1.0)

            p_flat=pred[b,t].flatten()
            t_flat=true[b,t].flatten()
            if np.std(p_flat)==0 or np.std(t_flat)==0:
                continue
            corr,_ = pearsonr(p_flat, t_flat)
            total_corr+=corr
    ssim_avg=ssim_total/(B*T)
    total_corr=total_corr/(B*T)


    return mae_loss, mse_loss,rmse_loss,ssim_avg,total_corr

if __name__ == '__main__':

    trues = np.load('results_npy/trues_restored.npy')
    preds = np.load('results_npy/preds_restored.npy')
    print(trues.shape, preds.shape)

    mae, mse, rmse,ssim1,corr= calculate_loss(preds, trues, clip_range=[0, 1])
    print(f"Test Results - MSE: {mse:.8f} | MAE: {mae:.8f} | RMSE: {rmse:.8f} | SSIM: {ssim1:.8f} | Corr: {corr:.8f}")