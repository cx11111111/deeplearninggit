import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

#读入数据
data=pd.read_csv(r"C:\Users\cx\Desktop\数据集\data\firstturbdata.csv",usecols=[12])
plt.plot(data)
plt.show()

#数据预处理
data=data.dropna()
data=data.astype("float32")
scaler=MinMaxScaler()
data=scaler.fit_transform(data)
print(data.shape)

#划分输入数据和标签
time_step=300 #用多少数据预测
pred=1   #预测多少
def create_data(data,time_step,pred):
    dataX,dataY=[],[]
    for i in range(time_step,len(data)-pred):
        dataX.append(data[i-time_step:i])
        dataY.append(data[i:i+pred])
    return np.array(dataX),np.array(dataY).reshape(-1,1)
dataX,dataY=create_data(data,time_step,pred)
print(dataX.shape)
print(dataY.shape)

#划分训练集、验证集和测试集
train_size=int(len(data)*0.7)
val_size=int(len(data)*0.85)
train_x=dataX[:train_size]
train_y=dataY[:train_size]
val_x=dataX[train_size:val_size]
val_y=dataY[train_size:val_size]
test_x=dataX[val_size:]
test_y=dataY[val_size:]

#把训练集、验证集和测试集转化为张量
train_x=torch.from_numpy(train_x).to(torch.float32)
train_y=torch.from_numpy(train_y).to(torch.float32)
val_x=torch.from_numpy(val_x).to(torch.float32)
val_y=torch.from_numpy(val_y).to(torch.float32)
test_x=torch.from_numpy(test_x).to(torch.float32)
test_y=torch.from_numpy(test_y).to(torch.float32)
print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(test_x.shape)
train_x=train_x.to('cuda:0')
train_y=train_y.to('cuda:0')
val_x=val_x.to('cuda:0')
val_y=val_y.to('cuda:0')

#构建模型
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class rnn_reg(torch.nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,out_size):
        super(rnn_reg,self).__init__()
        self.hidden_size1=hidden_size1
        self.hidden_size2=hidden_size2
        self.rnn1=torch.nn.RNN(input_size,hidden_size1,num_layers=1,batch_first=True)
        self.rnn2=torch.nn.RNN(hidden_size1,hidden_size2,num_layers=1,batch_first=True)
        self.reg=torch.nn.Linear(hidden_size2,out_size)

    def forward(self,x):
        h0 = torch.randn(1, x.size(0), self.hidden_size1,device=device).requires_grad_()
        out1, hn = self.rnn1(x, h0)
        h1 = torch.randn(1, x.size(0), self.hidden_size2,device=device).requires_grad_()
        out, hn = self.rnn2(out1, h1)
        out = self.reg(out[:, -1, :])
        return out.squeeze()

input_size=1
hidden_size1=32
hidden_size2=64
out_size=1
lr=0.00008
epochs=10
batch_size=32
train_data=TensorDataset(train_x,train_y)
train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=False)
net=rnn_reg(input_size,hidden_size1,hidden_size2,out_size)
net.to(device)
criterion=torch.nn.MSELoss()
optimizer=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=0.005)

#from torchinfo import summary
#print(summary(net,input_size=(175,5,1)))

for param_tensor in net.state_dict():
    print(param_tensor,net.state_dict()[param_tensor].size())

train_losses,val_losses=[],[]
for epoch in range(epochs):
    net=net.train()
    train_loss=0.0
    for inputs,labels in train_loader:
        out=net(inputs)
        loss=criterion(out,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*inputs.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    net = net.eval()
    val_out=net(val_x)
    val_loss=criterion(val_out,val_y)
    val_losses.append(val_loss.item())
    print('Epoch [{}/{}],Train Loss:{:.4f},Val Loss:{:.4f}'.format(epoch + 1, epochs, train_loss, val_loss))
plt.plot(train_losses,label='train loss')
plt.plot(val_losses,label='val loss')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#设置模型为评估模式
net=net.eval()
test_x=test_x.to(device)
pred_test=net(test_x)
pred_test=pred_test.data.cpu()

plt.plot(pred_test,'r',label='predict')
plt.plot(test_y,'green',label='real')
plt.legend(loc='best')
plt.show()

from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(test_y,pred_test)
print(MSE)