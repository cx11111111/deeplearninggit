import torch
from torch.nn import L1Loss
from argparse import ArgumentParser

# Device on which we port the model and variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loss function: L1loss (mean absolute error)
loss_func = L1Loss()

parser = ArgumentParser()
#训练模式或预测模式
parser.add_argument("--mode", type=str, choices=["train", "eval"],
                    default="train", help="Are we training or testing?")
#预训练网络的位置
parser.add_argument("--pretrained_path", type=str,
                    default="pretrained/car_sales.pt",
                    help="Location of the pretrained net")
#数据集路径
parser.add_argument("--dataset_path", type=str, default="data/monthly-car-sales.csv",
                    help="Location of the dataset file")
#使用多少时间步长的数据来预测
parser.add_argument("--lag", type=int, default=3,
                    help="Time lag used for preparing train and test X-Y pairs")
#训练集和测试集比例
parser.add_argument("--split_ratio", type=float, default=0.7,
                    help="Ratio for splitting the dataset into train-test subsets")
#LSTM隐藏状态向量维度
parser.add_argument("--hidden_dim", type=int, default=3,
                    help="Dimension of the LSTM hidden state vector")
#LSTM的层数
parser.add_argument("--num_layers", type=int, default=1,
                    help="Number of LSTM layers")
parser.add_argument("--epochs", type=int, default=1000,
                    help="Number of training epochs")
parser.add_argument("--bs", type=int, default=52,
                    help="Batch size")
parser.add_argument("--lr", type=float, default=5e-4,
                    help="Learning rate")
parser.add_argument("--wd", type=float, default=6e-12,
                    help="L2 regularization weight decay")

#存储参数
config = parser.parse_args()
