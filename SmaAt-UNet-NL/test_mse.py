import numpy as np
if __name__ == '__main__':
    trues = np.load('results_npy/trues.npy')
    preds = np.load('results_npy/preds.npy')
    # mse = MSE(preds,trues)
    # mae = MAE(preds,trues)
    # ssim = SSIM(preds,trues)
    # for i in range(output_length):
    #     preds=preds[:, i,:,:] 334 1 64 64
