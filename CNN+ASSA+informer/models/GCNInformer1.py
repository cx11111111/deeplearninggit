import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# LatentCorrelationLayer：生成潜在相关性，构建每个时间步之间的图结构，提供GCN的输入。
# GCNLayer：为所有涡轮机共享一个GCN层，这样可以在多个时间步上应用相同的图卷积操作。
# InformerModule：通过LSTM实现时间序列特征提取，最后输出一个时间步的预测结果。
# GCNInformer：整合了潜在相关性层、GCN层和Informer模块来处理时间序列预测任务。
# 潜在相关性层
class LatentCorrelationLayer(nn.Module):
    def __init__(self, num_turbines, embed_size):
        super(LatentCorrelationLayer, self).__init__()
        self.query_proj = nn.Linear(embed_size, embed_size)  # Qw
        self.key_proj = nn.Linear(embed_size, embed_size)  # Kw0
        self.embed_size = embed_size

    def forward(self, x):
        # x: [num_turbines, time_steps, embed_size]
        # Query and Key projections
        Qw = self.query_proj(x)
        Kw0 = self.key_proj(x)

        # Scaled dot-product attention
        attention_weights = torch.matmul(Qw, Kw0.transpose(-1, -2)) / (self.embed_size ** 0.5)
        attention_weights = F.softmax(attention_weights, dim=-1)

        # 生成相关性图结构 G0 = (Ww, B)
        correlated_x = torch.matmul(attention_weights, x)
        return correlated_x, attention_weights


# GCN层
class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x: [num_nodes, in_channels]
        x = self.gcn(x, edge_index)
        x = F.relu(x)
        return x


# Informer模块
class InformerModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InformerModule, self).__init__()
        self.lstm = nn.LSTM(input_dim, output_dim, batch_first=True)
        self.fc = nn.Linear(output_dim, 1)

    def forward(self, x):
        # x: [batch_size, time_steps, input_dim]
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 获取最后一个时间步的输出
        return out


# 整合的GCN-Informer模型
class GCNInformer(nn.Module):
    def __init__(self, num_turbines, time_steps, embed_size, gcn_out_channels, informer_out_dim):
        super(GCNInformer, self).__init__()
        self.latent_correlation_layer = LatentCorrelationLayer(num_turbines, embed_size)
        self.gcn_layer = GCNLayer(embed_size, gcn_out_channels)  # 单一GCN层
        self.informer = InformerModule(gcn_out_channels * num_turbines, informer_out_dim)
        self.fc = nn.Linear(informer_out_dim, 1)

    def forward(self, x, edge_index):
        # x: [num_turbines, time_steps, embed_size]
        # 1. 计算潜在相关性
        correlated_x, attention_weights = self.latent_correlation_layer(x)

        # 2. 使用GCN提取特征
        # 重新组织数据，适应GCN输入
        num_turbines, time_steps, embed_size = correlated_x.shape
        gcn_outputs = []
        for i in range(time_steps):
            timestep_x = correlated_x[:, i, :]  # 选择每个时间步的数据
            gcn_out = self.gcn_layer(timestep_x, edge_index)
            gcn_outputs.append(gcn_out)

        # 3. 将每个时间步的特征拼接
        gcn_outputs = torch.stack(gcn_outputs, dim=1)  # [num_turbines, time_steps, gcn_out_channels]

        # 4. 将GCN输出输入到Informer模块进行时间序列预测
        informer_out = self.informer(gcn_outputs.view(1, time_steps, -1))  # 展开后添加batch维度

        # 5. 全连接层获取最终输出
        output = self.fc(informer_out)
        return output, attention_weights


# 示例用法
num_turbines = 10
time_steps = 24
embed_size = 64
gcn_out_channels = 32
informer_out_dim = 64

model = GCNInformer(num_turbines, time_steps, embed_size, gcn_out_channels, informer_out_dim)
dummy_x = torch.randn(num_turbines, time_steps, embed_size)  # 模拟输入数据
dummy_edge_index = torch.randint(0, num_turbines, (2, 30))  # 图连接关系

output, attention_weights = model(dummy_x, dummy_edge_index)
print("Model output:", output)
print("Attention weights:", attention_weights)
