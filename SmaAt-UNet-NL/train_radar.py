# -*-coding:utf-8-*-
import os
from models.Unet import UNet
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tqdm
from tqdm import tqdm
from torch import optim
import time
import torch.nn.functional
import torch
from torch import nn
from API.radar_dataloader import load_data

import matplotlib.pyplot as plt
from models.SmaAt_UNet import SmaAt_UNet
# Hyperparameters


learning_rate = 0.001
epochs = 200
earlystopping = 15
save_every = 1

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def loss_func(y_pred, y_true):
    return nn.functional.mse_loss(y_pred, y_true, reduction="mean")

def fit(epochs, model, loss_func, opt, train_dl, valid_dl,
        dev=torch.device('cuda'), save_every: int = None, tensorboard: bool = False, earlystopping=None,
        lr_scheduler=None, start_epoch=0, best_val_loss=float('inf')):
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(comment=f"{model.__class__.__name__}")

    train_losses = []
    val_losses = []
    earlystopping_counter = 0

    for epoch in tqdm(range(start_epoch, start_epoch + epochs), desc="Epochs", leave=True):
        model.train()
        train_loss = 0.0
        for i, (xb, yb) in enumerate(tqdm(train_dl, desc="Batches", leave=False)):
            xb = xb.squeeze()
            yb = yb.squeeze()
            loss = loss_func(model(xb.to(dev)), yb.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)
        train_losses.append(train_loss)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, (xb, yb) in enumerate(tqdm(valid_dl, desc="Validation", leave=False)):
                xb = xb.squeeze()
                yb = yb.squeeze()
                y_pred = model(xb.to(dev))
                loss = loss_func(y_pred, yb.to(dev))
                val_loss += loss.item()

                # Visualization logic
                if i == 0:  # Visualize the first batch
                    visualize_predictions(y_pred.cpu(), yb.cpu(), epoch, i)

            val_loss /= len(valid_dl)
        val_losses.append(val_loss)

        # Save the model with the best validation loss
        save_path = "TianChi/Unet"
        os.makedirs(save_path, exist_ok=True)  # Ensure save path exists
        if val_loss < best_val_loss:
            torch.save({
                'model': model,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, f"{save_path}/best_val_loss_{model.__class__.__name__}.pt")
            best_val_loss = val_loss
            earlystopping_counter = 0
        else:
            if earlystopping is not None:
                earlystopping_counter += 1
                if earlystopping_counter >= earlystopping:
                    print(f"Stopping early --> val_loss has not decreased over {earlystopping} epochs")
                    break

        print(f"Epoch: {epoch}, Train_loss: {train_loss:.6f}, Val_loss: {val_loss:.6f}, "
              f"lr: {get_lr(opt):.6f}, Early stopping counter: {earlystopping_counter}/{earlystopping}" if earlystopping else "")

        if tensorboard:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Parameters/learning_rate', get_lr(opt), epoch)

        if save_every and epoch % save_every == 0:
            torch.save({
                'model': model,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, f"{save_path}/model_{model.__class__.__name__}_epoch_{epoch}_val_loss{val_loss:.6f}.pt")

        if lr_scheduler:
            lr_scheduler.step(val_loss)

    if tensorboard:
        writer.close()

def visualize_predictions(y_pred, y_true, epoch, batch_idx):
    """
    Visualizes predictions and ground truth for a batch as sequences.
    Saves each sample's sequences (predicted and ground truth) in one image.
    """
    save_dir = f"visualizations/epoch_{epoch}_batch_{batch_idx}"
    os.makedirs(save_dir, exist_ok=True)

    num_samples = min(5, y_pred.shape[0])  # Visualize up to 4 samples
    sequence_length = y_pred.shape[1]     # Number of frames in the sequence

    for sample_idx in range(num_samples):
        plt.figure(figsize=(20, 8))  # Adjust size to fit the sequences

        for frame_idx in range(sequence_length):
            # Plot predicted frames
            plt.subplot(2, sequence_length, frame_idx + 1)
            plt.imshow(y_pred[sample_idx, frame_idx].cpu().numpy(), cmap='gray')
            plt.title(f"pred - Frame {frame_idx}")
            plt.axis("off")

            # Plot ground truth frames
            plt.subplot(2, sequence_length, frame_idx + 1 + sequence_length)
            plt.imshow(y_true[sample_idx, frame_idx].cpu().numpy(), cmap='gray')
            plt.title(f"target- Frame {frame_idx}")
            plt.axis("off")

        plt.suptitle(f"Sample {sample_idx} - Predicted vs Ground Truth", fontsize=16)
        save_path = os.path.join(save_dir, f"sample_{sample_idx}_sequences.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = UNet(12, 12)
    model.to(dev)

    opt = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=4)

    # Load pretrained weights if available
    checkpoint_path = r""
    # checkpoint_path = ''
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"Resumed training from epoch {start_epoch} with best validation loss {best_val_loss:.6f}")

    train_loader, val_loader,_,_, _ = load_data(
        batch_size=64, val_batch_size=64, data_root='data/power_matrices.npy', num_workers=0,
        seq_len=24, step=6, max_value=1567.02, target_shape=(64, 64), train_val_split=0.9, seed=42)


    fit(epochs, model, loss_func, opt, train_loader, val_loader, dev, save_every=save_every,
        tensorboard=True, earlystopping=earlystopping, lr_scheduler=lr_scheduler,
        start_epoch=start_epoch, best_val_loss=best_val_loss)
