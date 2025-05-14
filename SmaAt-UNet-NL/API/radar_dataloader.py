import numpy as np
import torch
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, data, seq_len=24, step=1, max_value=1567.02, target_shape=(64, 64)):
        # 输入数据
        self.data = data
        self.seq_len = seq_len  # 序列长度
        self.step = step  # 滑动步长
        self.max_value = max_value  # 最大值归一化
        self.target_shape = target_shape  # 输出的目标形状

        # 预处理数据
        self.data[self.data < 0] = 0  # 负值替换为0
        self.data = self.data / self.max_value  # 最大值归一化

        # 计算滑窗数量
        self.num_windows = (data.shape[0] - seq_len) // step + 1

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start_idx = idx * self.step
        end_idx = start_idx + self.seq_len

        # 获取输入和真实值
        input_seq = self.data[start_idx:start_idx + 12]  # 前12帧为输入
        target_seq = self.data[start_idx + 12:start_idx + 24]  # 后12帧为真实值

        # 填充数据以满足目标形状
        # 假设填充是按目标形状来处理的，可能会涉及扩展维度
        # 填充到 (12, 64, 64)，这里假设在每个轴的末尾填充0

        input_seq_padded = np.pad(input_seq, ((0, 0), (0, self.target_shape[0] - input_seq.shape[1]), (0, self.target_shape[1] - input_seq.shape[2])), mode='constant', constant_values=0)
        target_seq_padded = np.pad(target_seq, ((0, 0), (0, self.target_shape[0] - target_seq.shape[1]), (0, self.target_shape[1] - target_seq.shape[2])), mode='constant', constant_values=0)

        # 转为torch tensor
        input_tensor = torch.tensor(input_seq_padded, dtype=torch.float32)
        target_tensor = torch.tensor(target_seq_padded, dtype=torch.float32)

        return input_tensor, target_tensor


import torch
import numpy as np
import random
from torch.utils.data import random_split, DataLoader

def load_data(
        batch_size, val_batch_size,
        data_root, num_workers,
        seq_len=24, step=12, max_value=1567.02, target_shape=(64, 64), train_val_split=0.9, seed=42):

    # 固定随机数种子，确保结果可复现
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # 如果使用CUDA的话
    # torch.backends.cudnn.deterministic = True  # 对cuDNN的设置，确保结果可复现
    # torch.backends.cudnn.benchmark = False  # 禁用优化，避免与硬件配置有关的随机性

    # 加载你的数据集
    data = np.load(data_root)  # 假设你已经加载了数据，形状为 (35280, 27, 7)

    # 创建SlidingWindowDataset对象
    full_dataset = SlidingWindowDataset(data, seq_len=seq_len, step=step, max_value=max_value, target_shape=target_shape)

    # 按照train_val_split比例划分训练集和验证集
    train_size = int(train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # 创建dataloader
    dataloader_train = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_validation = DataLoader(
        val_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_test = DataLoader(
        val_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std

if __name__ == '__main__':

    dataloader_train, dataloader_validation, dataloader_test, mean, std = load_data(
        batch_size=64, val_batch_size=64, data_root='/path/to/data/power_matrices.npy', num_workers=0,
        seq_len=24, step=12, max_value=1567.02, target_shape=(64, 64), train_val_split=0.9, seed=42)

