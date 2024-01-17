import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader,TensorDataset

data=pd.read_csv(r"C:\Users\22279\Desktop\数据集\wind_dataset.csv",usecols=[1])
plt.plot(data)
plt.show()
#数据预处理
data=data.dropna()
data=data.values.astype("float32")
scaler=MinMaxScaler()
data=scaler.fit_transform(data)
print(data.shape)

#划分输入数据和标签
time_step=1 #用多少数据预测
pred=1 #预测长度
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
train_size=int(len(dataX)*0.7)
val_size=int(len(dataX)*0.85)
train_x=dataX[:train_size]
val_x=dataX[train_size:val_size]
test_x=dataX[val_size:]
train_y=dataY[:train_size]
val_y=dataY[train_size:val_size]
test_y=dataY[val_size:]
#转化为张量
train_x=torch.from_numpy(train_x).to(torch.float32)
val_x=torch.from_numpy(val_x).to(torch.float32)
test_x=torch.from_numpy(test_x).to(torch.float32)
train_y=torch.from_numpy(train_y).to(torch.float32)
val_y=torch.from_numpy(val_y).to(torch.float32)
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
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LSTMModel(torch.nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,hidden_size3,output_size):
        super(LSTMModel,self).__init__()
        self.hidden_size1=hidden_size1
        self.hidden_size2=hidden_size2
        self.hidden_size3=hidden_size3
        self.lstm1=torch.nn.LSTM(input_size,hidden_size1,batch_first=True,num_layers=1)
        self.lstm2=torch.nn.LSTM(hidden_size1,hidden_size2,batch_first=True,num_layers=1)
        self.lstm3=torch.nn.LSTM(hidden_size2,hidden_size3,batch_first=True,num_layers=1)
        self.linear=torch.nn.Linear(hidden_size3,output_size)
        self.dropout=torch.nn.Dropout(p=0.3)

    def forward(self,x):
        h0_1=torch.zeros(1,x.size(0),self.hidden_size1,device=device).requires_grad_()
        c0_1=torch.zeros(1,x.size(0),self.hidden_size1,device=device).requires_grad_()
        out,(hn_1,cn_1)=self.lstm1(x,(h0_1,c0_1))
        h0_2=torch.zeros(1,x.size(0),self.hidden_size2,device=device).requires_grad_()
        c0_2=torch.zeros(1,x.size(0),self.hidden_size2,device=device).requires_grad_()
        out,(hn_2,cn_2)=self.lstm2(out,(h0_2,c0_2))
        h0_3=torch.zeros(1,x.size(0),self.hidden_size3,device=device).requires_grad_()
        c0_3=torch.zeros(1,x.size(0),self.hidden_size3,device=device).requires_grad_()
        out,(hn_3,cn_3)=self.lstm3(out,(h0_3,c0_3))
        out=self.dropout(out[:,-1,:])
        out=self.linear(out)
        return out.squeeze()

#定义超参数
input_size=1
hidden_size1=64
hidden_size2=128
hidden_size3=32
output_size=1
learning_rate=0.001
num_epochs=10
batch_size=64
train_data=TensorDataset(train_x,train_y)
train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=False)
model=LSTMModel(input_size,hidden_size1,hidden_size2,hidden_size3,output_size)
model.to(device)
criterion=torch.nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.00015)


#from torchinfo import summary
#print(summary(net,input_size=(175,5,1)))

for param_tensor in model.state_dict():
    print(param_tensor,model.state_dict()[param_tensor].size())

#训练模型
train_losses,val_losses=[],[]
for epoch in range(num_epochs):
    train_loss=0.0
    model.train()
    for inputs,labels in train_loader:
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*inputs.size(0)
    #计算平均损失
    train_loss/=len(train_loader.dataset)
    train_losses.append(train_loss)

    #评估模型
    model.eval()
    with torch.no_grad():
        val_outputs=model(val_x)
        val_loss=criterion(val_outputs,val_y)
    val_losses.append(val_loss.item())
    print('Epoch [{}/{}],Train Loss:{:.4f},Val Loss:{:.4f}'.format(epoch+1,num_epochs,train_loss,val_loss))

plt.plot(train_losses,label='train loss')
plt.plot(val_losses,label='val loss')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

model=model.eval()
test_x=test_x.to(device)
prediction=model(test_x)
prediction=prediction.data.cpu()

plt.plot(prediction,label='predict')
plt.plot(test_y,label='real')
plt.legend(loc='best')
plt.show()

