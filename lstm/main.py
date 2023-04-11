import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# 读入数据
data=pd.read_csv(r"C:\Users\cx\Desktop\数据集\data\firstturbdata.csv",usecols=[12])
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
print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(test_x.shape)

#建立神经网络模型
model = keras.Sequential()
model.add(layers.LSTM(64, return_sequences=True, input_shape=(train_x.shape[1:])))
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.LSTM(32))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1))

model.compile(optimizer=keras.optimizers.Adam(), loss='mse',metrics=['accuracy'])
learning_rate= keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.7, min_lr=0.00000001)
#显示模型结构
model.summary()

# 训练模型
train = model.fit(train_x, train_y,batch_size = 128,epochs=10,validation_data=(val_x, val_y),callbacks=[learning_rate])

# loss变化趋势可视化
plt.plot(train.history['loss'],label='training loss')
plt.plot(train.history['val_loss'], label='val loss')
plt.legend(loc='upper right')
plt.show()

# 输入测试数据,输出预测结果
y_pred = model.predict(test_x)

# 反归一化
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(test_y)

# 预测效果可视化
plt.plot(y_test,label='real')
plt.plot(y_pred,label='prediction')
plt.legend(loc='upper right')
plt.show()