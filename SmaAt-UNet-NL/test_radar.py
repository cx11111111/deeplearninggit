import os
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm

from API.radar_dataloader import load_data
# from API.dataloader_moving_mnist import load_data
# from API.vis_valid import save_visualization
from models.SmaAt_UNet import SmaAt_UNet
from models.Unet import UNet

train_loader, val_loader,_,_, _ = load_data(
        batch_size=64, val_batch_size=64, data_root='data/power_matrices.npy', num_workers=0,
        seq_len=24, step=12, max_value=1567.02, target_shape=(64, 64), train_val_split=0.9, seed=42)

def restore(data,original_shape=(27,7)):
    h,w=original_shape
    return data[:,:,:h,:w]

def denormalize(data,max_value=1567.02):
    data_denorm=data*max_value
    return data_denorm

def get_npy():
    # Initialize model and load checkpoint
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = UNet(12, 12).to(dev)
    checkpoint_path = 'TianChi/Unet/best_val_loss_UNet.pt'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Initialize storage for predictions and ground truths
    trues_lst, preds_lst = [], []

    # Inference
    with torch.no_grad():
        for i,(xb,yb) in enumerate(tqdm(val_loader, leave=False, desc="Processing Validation Data")):
            '''            
            B,T,C,H,W  lstm , C=1 C=3 C=4 1km 5km 10km B,T,C,H,W---> B,T,C',H,W           
            B,C,H,W Unet B,C,H,W --->B,C',H,W H/2 4 8             
            '''
            xb = xb.squeeze()
            yb = yb.squeeze()
            xb, yb = xb.to(dev), yb.to(dev)
            y_pred = model(xb)

            # Collect predictions and true values
            trues_lst.append(yb.cpu().numpy())
            preds_lst.append(y_pred.cpu().numpy())
            # if i == 0:  # 可调整条件控制保存可视化的频率
            #     save_visualization(yb.cpu().numpy(), y_pred.cpu().numpy(),
            #                        r'E:\personal_reserch\Unet_Radar_GAN\results', i)

    # Concatenate collected data
    trues = np.concatenate(trues_lst, axis=0).squeeze()
    preds = np.concatenate(preds_lst, axis=0).squeeze()
    # Save results
    folder_path = 'results_npy'
    os.makedirs(folder_path, exist_ok=True)

    np.save(osp.join(folder_path, 'trues.npy'), trues)
    np.save(osp.join(folder_path, 'preds.npy'), preds)

    #还原为27*7尺寸
    trues_restored=restore(trues,original_shape=(27,7))
    preds_restored=restore(preds,original_shape=(27,7))
    np.save(osp.join(folder_path, 'trues_restored.npy'), trues_restored)
    np.save(osp.join(folder_path, 'preds_restored.npy'), preds_restored)

    #反归一化
    trues_denorm=denormalize(trues_restored, max_value=1567.02)
    preds_denorm=denormalize(preds_restored, max_value=1567.02)
    np.save(osp.join(folder_path, 'trues_denorm.npy'), trues_denorm)
    np.save(osp.join(folder_path, 'preds_denorm.npy'), preds_denorm)

    print(f"保存完成：trues shape: {trues.shape}, preds shape: {preds.shape}")
    print(f"还原后 shape: {trues_restored.shape}, {preds_restored.shape}")
    print(f"反归一化后 shape: {trues_denorm.shape}, {preds_denorm.shape}")

if __name__ == '__main__':
    get_npy()
