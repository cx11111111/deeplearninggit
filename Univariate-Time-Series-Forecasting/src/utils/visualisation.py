from globals import *

import torch
import numpy as np
import matplotlib.pyplot as plt


x_ticks = list()
tick_positions = list()


def show_evaluation(net, dataset, scaler, debug=True):
    ''' 评估 RNN 在整个数据集上的性能，并显示预测值和目标值.
    参数:
        net (nn.Module): RNN to evaluate
        dataset (numpy.ndarray): dataset
        scaler (MinMaxScaler): 反归一化
        debug (bool): should we calculate/display eval.MSE/MAE
    '''
    dataset = torch.FloatTensor(dataset).unsqueeze(-1).to(device)
    total_train_size = int(config.split_ratio * len(dataset))

    # 预测整个数据集
    net.eval()
    test_predict = net(dataset)

    # 对实际值和预测值反归一化
    test_predict = scaler.inverse_transform(test_predict.cpu().data.numpy())
    dataset = scaler.inverse_transform(dataset.cpu().squeeze(-1).data.numpy())

    # 绘制原始序列与预测序列
    plt.figure(figsize=(8, 5))
    plt.axvline(x=total_train_size, c='r')
    plt.plot(dataset)
    plt.plot(test_predict)
    plt.xticks(tick_positions, x_ticks, size='small')
    plt.xlabel('Year-Month')
    plt.ylabel("Number of car sales")
    plt.title('Univariate Time-Series Forecast')
    plt.legend(['Train-Test split', 'Target', 'Prediction'])
    plt.show()

    if debug:
        # 计算整个数据集的MSE、MAE
        total_mse = (np.square(test_predict - dataset)).mean()
        total_mae = (np.abs(test_predict - dataset)).mean()
        #计算训练集的MSE、MAE
        train_mse = (np.square(test_predict - dataset)
                     )[:total_train_size].mean()
        train_mae = (np.abs(test_predict - dataset))[:total_train_size].mean()
        #计算测试集的MSE、MAE
        test_mse = (np.square(test_predict - dataset)
                    )[total_train_size:].mean()
        test_mae = (np.abs(test_predict - dataset))[total_train_size:].mean()

        print(f"Total MSE:  {total_mse:.4f}  |  Total MAE:  {total_mae:.4f}")
        print(f"Train MSE:  {train_mse:.4f}  |  Train MAE:  {train_mae:.4f}")
        print(f"Test MSE:   {test_mse:.4f}  |  Test MAE:   {test_mae:.4f}")


def show_loss(history):
    ''' Display train and evaluation loss

    Arguments:
        history(dict): Contains train and test loss logs
    '''
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train loss')
    plt.plot(history['test_loss'], label='Evaluation loss')

    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()


def display_dataset(dataset, xlabels):
    ''' Displays the loaded data

    Arguments:
        dataset(numpy.ndarray): loaded data
        xlabels(numpy.ndarray): strings representing according dates
    '''
    global x_ticks
    global tick_positions
    # 我们无法在 x 轴上显示数据集中的每个日期，因为我们无法清楚地看到任何标签。所以我们提取每个第 n 个标签/刻度
    segment = int(len(dataset) / 6)

    for i, date in enumerate(xlabels):
        if i > 0 and (i + 1) % segment == 0:
            x_ticks.append(date)
            tick_positions.append(i)
        elif i == 0:
            x_ticks.append(date)
            tick_positions.append(i)

    # Display loaded data
    plt.figure(figsize=(8, 5))
    plt.plot(dataset)
    plt.title('Monthly car sales')
    plt.xlabel('Year-Month')
    plt.ylabel("Number of car sales")
    plt.xticks(tick_positions, x_ticks, size='small')
    plt.show()
