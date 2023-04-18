from globals import *

import torch
import numpy as np
import matplotlib.pyplot as plt


x_ticks = list()
tick_positions = list()


def show_evaluation(net, dataset, scaler, debug=True):
    ''' 评估 RNN 在测试集上的性能，并显示预测值和目标值.
    参数:
        net (nn.Module): RNN to evaluate
        dataset (numpy.ndarray): dataset
        scaler (MinMaxScaler): 反归一化
        debug (bool): should we calculate/display eval.MSE/MAE
    '''
    dataset = torch.FloatTensor(dataset).unsqueeze(-1).to(device)
    total_train_size = int(config.split_ratio1 * len(dataset))
    test_set=dataset[total_train_size:]

    # 预测测试集
    net.eval()
    test_predict = net(test_set)

    # 对实际值和预测值反归一化
    test_predict = scaler.inverse_transform(test_predict.cpu().data.numpy())
    test_set = scaler.inverse_transform(test_set.cpu().squeeze(-1).data.numpy())

    # 绘制原始序列与预测序列
    plt.plot(test_set,label='real')
    plt.plot(test_predict,label='predict')
    plt.ylabel("Patv")
    plt.title('Forecast and Real')
    plt.legend()
    plt.show()

    if debug:
        # 计算整个数据集的MSE、MAE
        #total_mse = (np.square(test_predict - dataset)).mean()
        #total_mae = (np.abs(test_predict - dataset)).mean()
        #计算训练集的MSE、MAE
        #train_mse = (np.square(test_predict - dataset))[:total_train_size].mean()
        #train_mae = (np.abs(test_predict - dataset))[:total_train_size].mean()
        #计算测试集的MSE、MAE
        test_mse = (np.square(test_predict - test_set)).mean()
        test_mae = (np.abs(test_predict - test_set)).mean()

        #print(f"Total MSE:  {total_mse:.4f}  |  Total MAE:  {total_mae:.4f}")
        #print(f"Train MSE:  {train_mse:.4f}  |  Train MAE:  {train_mae:.4f}")
        print(f"Test MSE:   {test_mse:.4f}  |  Test MAE:   {test_mae:.4f}")


def show_loss(history):
    ''' Display train and evaluation loss

    Arguments:
        history(dict): Contains train and test loss logs
    '''
    plt.plot(history['train_loss'], label='Train loss')
    plt.plot(history['val_loss'], label='Val loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def display_dataset(dataset):
    ''' Displays the loaded data

    Arguments:
        dataset(numpy.ndarray): loaded data
        xlabels(numpy.ndarray): strings representing according dates
    '''
    #global x_ticks
    #global tick_positions
    # 我们无法在 x 轴上显示数据集中的每个日期，因为我们无法清楚地看到任何标签。所以我们提取每个第 n 个标签/刻度
    #segment = int(len(dataset) / 6)

    #for i, date in enumerate(xlabels):
        #if i > 0 and (i + 1) % segment == 0:
            #x_ticks.append(date)
            #tick_positions.append(i)
        #elif i == 0:
            #x_ticks.append(date)
            #tick_positions.append(i)

    # Display loaded data
    plt.plot(dataset)
    plt.title('TurbData')
    plt.ylabel("Patv")
    #plt.xticks(tick_positions, x_ticks, size='small')
    plt.legend()
    plt.show()
