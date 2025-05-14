import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class QuantileRegressionModel(nn.Module):
    def __init__(self):
        super(QuantileRegressionModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

def quantile_loss(y_pred, y_true, quantile):
    errors = y_true - y_pred
    loss = torch.max((quantile - 1) * errors, quantile * errors).mean()
    return loss

# 示例训练代码
quantile_levels = [0.25, 0.5, 0.75]  # 25%, 50%, 75% 分位点
models = [QuantileRegressionModel() for _ in quantile_levels]

for quantile, model in zip(quantile_levels, models):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = quantile_loss(y_pred, y, quantile)
        loss.backward()
        optimizer.step()

# 预测结果为多个分位点
pred_25 = models[0](x)
pred_50 = models[1](x)
pred_75 = models[2](x)

