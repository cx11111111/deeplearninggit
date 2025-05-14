import torch
import numpy as np
import pandas as pd
from models import CNNInformer
from utils.timefeatures import time_features
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 分位数损失函数
def quantile_loss(q, y_true, y_pred):
    """
    q: 分位数 (0 < q < 1)
    y_true: 真实值 (torch tensor)
    y_pred: 预测值 (torch tensor)
    """
    e = y_true - y_pred
    return torch.mean(torch.max(q * e, (q - 1) * e))



def tslib_data_loader(window, length_size, batch_size, data, data_mark):
    """
    数据加载器函数，用于加载和预处理时间序列数据，以用于训练模型。

    仅仅适用于 多变量预测多变量（可以单独取单变量的输出），或者单变量预测单变量
    也就是y里也会有外生变量？？

    参数:
    - window: 窗口大小，用于截取输入序列的长度。
    - length_size: 目标序列的长度。
    - batch_size: 批量大小，决定每个训练批次包含的数据样本数量。
    - data: 输入时间序列数据。
    - data_mark: 输入时间序列的数据标记，用于辅助模型训练或增加模型的多样性。

    返回值:
    - dataloader: 数据加载器，用于批量加载处理后的训练数据。
    - x_temp: 处理后的输入数据。
    - y_temp: 处理后的目标数据。
    - x_temp_mark: 处理后的输入数据的标记。
    - y_temp_mark: 处理后的目标数据的标记。
    """

    # 构建模型的输入
    seq_len = window
    sequence_length = seq_len + length_size
    result = np.array([data[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])
    result_mark = np.array([data_mark[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])

    # 划分x与y
    x_temp = result[:, :-length_size]
    y_temp = result[:, -(length_size + int(window / 2)):]

    x_temp_mark = result_mark[:, :-length_size]
    y_temp_mark = result_mark[:, -(length_size + int(window / 2)):]

    # 转换为Tensor和数据类型
    x_temp = torch.tensor(x_temp).type(torch.float32)
    x_temp_mark = torch.tensor(x_temp_mark).type(torch.float32)
    y_temp = torch.tensor(y_temp).type(torch.float32)
    y_temp_mark = torch.tensor(y_temp_mark).type(torch.float32)

    ds = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dataloader, x_temp, y_temp, x_temp_mark, y_temp_mark



df = pd.read_csv(r'C:\Users\22279\deeplearninggit\CNN+ASSA+informer\data\combined_results2.csv')
data_dim = df[df.columns.drop('date')].shape[1]  # 一共多少个变量 这个不去动
data_target = df['Patv']  # 预测的目标变量 把预测的列名改为target
data = df[df.columns.drop('date')]  # 选取所有的数据
# 时间戳
df_stamp = df[['date']]
df_stamp['date'] = pd.to_datetime(df_stamp.date)
data_stamp = time_features(df_stamp, timeenc=1, freq='T')  # 这一步很关键，注意数据的freq 你的数据是最小精度是什么就填什么，下面有

scaler = MinMaxScaler()
data_inverse = scaler.fit_transform(np.array(data))

data_length = len(data_inverse)
train_set = 0.7
val_set=0.85

data_train = data_inverse[:int(train_set * data_length), :]  # 读取目标数据，第一列记为0：1，后面以此类推, 训练集和验证集，如果是多维输入的话最后一列为目标列数据
data_train_mark = data_stamp[:int(train_set * data_length), :]
data_val = data_inverse[int(train_set * data_length):int(val_set * data_length), :]
data_val_mark = data_stamp[int(train_set * data_length):int(val_set * data_length), :]
data_test = data_inverse[int(val_set * data_length):, :]  # 这里把训练集和测试集分开了，也可以换成两个csv文件
data_test_mark = data_stamp[int(val_set * data_length):, :]

n_feature = data_dim

window = 6# 模型输入序列长度
length_size = 1  # 预测结果的序列长度
batch_size = 32

train_loader, x_train, y_train, x_train_mark, y_train_mark = tslib_data_loader(window, length_size, batch_size, data_train, data_train_mark)
val_loader, x_val, y_val, x_val_mark, y_val_mark = tslib_data_loader(window, length_size, batch_size, data_val, data_val_mark)
test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(window, length_size, batch_size, data_test, data_test_mark)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 10 # 训练迭代次数
learning_rate = 0.0001  # 学习率
scheduler_patience = int(0.25 * num_epochs)  # 转换为整数  学习率调整的patience
early_patience = 0.2  # 训练迭代的早停比例 即patience=0.25*num_epochs

class Config:
    def __init__(self):
        # basic
        self.seq_len = window  # input sequence length
        self.label_len = int(window / 2)  # start token length
        self.pred_len = length_size  # 预测序列长度
        self.freq = 't'  # 时间的频率，
        # 模型训练
        self.batch_size = batch_size  # 批次大小
        self.num_epochs = num_epochs  # 训练的轮数
        self.learning_rate = learning_rate  # 学习率
        self.stop_ratio = early_patience  # 早停的比例
        # 模型 define
        self.dec_in = data_dim  # 解码器输入特征数量, 输入几个变量就是几
        self.enc_in = data_dim  # 编码器输入特征数量
        self.c_out = 1  # 输出维度##########这个很重要
        # 模型超参数
        self.d_model = 32  # 模型维度
        self.n_heads = 8  # 多头注意力头数
        self.dropout = 0.1  # 丢弃率
        self.e_layers = 3  # 编码器块的数量
        self.d_layers = 3  # 解码器块的数量
        self.d_ff = 64  # 全连接网络维度
        self.factor = 5  # 注意力因子
        self.activation = 'gelu'  # 激活函数
        self.channel_independence = 0  # 频道独立性，0:频道依赖，1:频道独立

        self.top_k = 5  # TimesBlock中的参数
        self.num_kernels = 6  # Inception中的参数
        self.distil = 1  # 是否使用蒸馏，1为True
        # 一般不需要动的参数
        self.embed = 'timeF'  # 时间特征编码方式
        self.output_attention = 0  # 是否输出注意力
        self.task_name = 'short_term_forecast'  # 模型的任务，一般不动但是必须这个参数


config = Config()
CNNInformer.Model(config).to(device)
# 模型定义




quantiles = [0.025,0.125,0.25,0.375,0.625,0.75,0.875, 0.975]
models = {}
optimizers = {}

for q in quantiles:
    models[q] = CNNInformer.Model(config).to(device)
    optimizers[q] = torch.optim.Adam(models[q].parameters(), lr=0.01)

# 训练
epochs = 10
for q in quantiles:
    model = models[q]
    optimizer = optimizers[q]

    for epoch in range(epochs):
        model.train()
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(train_loader):
            datapoints, labels, datapoints_mark, labels_mark = datapoints.to(device), labels.to(
                device), datapoints_mark.to(device), labels_mark.to(device)
            optimizer.zero_grad()
            y_pred = model(datapoints, datapoints_mark, labels, labels_mark, None).squeeze()
            labels = labels[:, -length_size:].squeeze()
            loss = quantile_loss(q, labels, y_pred)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Quantile {q}, Epoch {epoch+1}, Train Loss: {loss.item()}")


        with (torch.no_grad()):  # 关闭自动求导以节省内存和提高效率
            for val_x, val_y, val_x_mark, val_y_mark in val_loader:
                val_x, val_y, val_x_mark, val_y_mark = val_x.to(device), val_y.to(device), val_x_mark.to(device), val_y_mark.to(device)  # 将数据移到GPU
                pred_val_y = model(val_x, val_x_mark, val_y, val_y_mark, None).squeeze()
                val_y = val_y[:, -length_size:].squeeze()  # 注意这一步
                loss = quantile_loss(q, val_y, pred_val_y)
            if epoch % 10 == 0:
                print(f"Quantile {q}, Epoch {epoch}, Val Loss: {loss.item()}")
    model.eval()


# 预测
y_lower_pred95 = models[0.025](x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device)).detach().cpu()
y_upper_pred95 = models[0.975](x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device)).detach().cpu()

y_lower_pred75 = models[0.125](x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device)).detach().cpu()
y_upper_pred75 = models[0.875](x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device)).detach().cpu()

y_lower_pred50 = models[0.25](x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device)).detach().cpu()
y_upper_pred50 = models[0.75](x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device)).detach().cpu()

y_lower_pred25 = models[0.375](x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device)).detach().cpu()
y_upper_pred25 = models[0.625](x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device)).detach().cpu()

y_true = y_test[:,-length_size:,-1:].detach().cpu()

# 可能需要调整pred和true的维度，使其变为二维数组
y_true = y_true[:, :, -1]
y_lower_pred95 = y_lower_pred95[:, :, -1]
y_upper_pred95 = y_upper_pred95[:, :, -1]
y_lower_pred75 = y_lower_pred75[:, :, -1]
y_upper_pred75 = y_upper_pred75[:, :, -1]
y_lower_pred50 = y_lower_pred50[:, :, -1]
y_upper_pred50 = y_upper_pred50[:, :, -1]
y_lower_pred25 = y_lower_pred25[:, :, -1]
y_upper_pred25 = y_upper_pred25[:, :, -1]

print("Shape of pred after adjustment:", y_true.shape,y_upper_pred95.shape)

#  #这段代码是为了重新更新scaler，因为之前定义的scaler是是十六维，这里重新根据目标数据定义一下scaler
# y_data_test_inverse = scaler.fit_transform(np.array(data_target).reshape(-1, 1))
# pred_uninverse = scaler.inverse_transform(pred[:, -1:])    #如果是多步预测， 选取最后一列
# true_uninverse = scaler.inverse_transform(true[:, -1:])
#
# true, pred = true_uninverse, pred_uninverse
# print(true.shape,pred.shape)

# 绘图
plt.figure()
plt.plot(range(300), y_true[100:400,:], color="red", label="True values", alpha=0.6)
plt.fill_between(range(300), y_lower_pred95[100:400,:].squeeze(), y_upper_pred95[100:400,:].squeeze(), color="green", alpha=0.1, label="95% prediction interval")
plt.fill_between(range(300), y_lower_pred75[100:400,:].squeeze(), y_upper_pred75[100:400,:].squeeze(), color="green", alpha=0.3, label="75% prediction interval")
plt.fill_between(range(300), y_lower_pred50[100:400,:].squeeze(), y_upper_pred50[100:400,:].squeeze(), color="green", alpha=0.5, label="50% prediction interval")
plt.fill_between(range(300), y_lower_pred25[100:400,:].squeeze(), y_upper_pred25[100:400,:].squeeze(), color="green", alpha=0.7, label="25% prediction interval")
plt.legend()
plt.title("Prediction Interval with Quantile Regression")
plt.show()

