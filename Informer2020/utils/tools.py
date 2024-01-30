import numpy as np
import torch

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = np.mean(data,axis=0)
        self.std = np.std(data,axis=0)

    def transform(self, data):
        if torch.is_tensor(data):
            mean = torch.as_tensor(self.mean, dtype=data.dtype, device=data.device)
            std = torch.as_tensor(self.std, dtype=data.dtype, device=data.device)
        else:
            mean = self.mean
            std = self.std
        return (data - mean) / (std + 1e-8)

    def inverse_transform(self, data):
        if torch.is_tensor(data):
            mean = torch.as_tensor(self.mean, dtype=data.dtype, device=data.device)
            std = torch.as_tensor(self.std, dtype=data.dtype, device=data.device)
        else:
            mean = self.mean
            std = self.std

        # Adjust mean and std shapes for broadcasting if necessary
        if data.ndim > 1 and (mean.ndim < data.ndim or std.ndim < data.ndim):
            shape_diff = data.ndim - mean.ndim
            mean = mean.reshape(*([1] * shape_diff), *mean.shape)
            std = std.reshape(*([1] * shape_diff), *std.shape)

        return (data * std) + mean

