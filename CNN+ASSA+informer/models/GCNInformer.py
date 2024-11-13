import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# LatentCorrelationLayer：使用自注意力机制来计算潜在相关性。通过Query和Key生成注意力权重，用于捕获不同风力涡轮机之间的依赖关系。
# GCNLayer：每个风力涡轮机的数据通过一个独立的GCN层提取图结构特征。GCN层使用拉普拉斯矩阵，将数据通过图卷积方式进行信息聚合。
# InformerModule：使用LSTM来模拟Informer的序列预测功能。虽然本例中用的是标准LSTM，但可以根据需求进一步实现稀疏注意力机制来优化时间复杂度。
# 输出：该模型返回最终预测结果和注意力权重矩阵，以便进一步分析不同时间步的依赖关系。
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


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x: [num_nodes, in_channels]
        x = self.gcn(x, edge_index)
        x = F.relu(x)
        return x


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


class GCNInformer(nn.Module):
    def __init__(self, num_turbines, time_steps, embed_size, gcn_out_channels, informer_out_dim):
        super(GCNInformer, self).__init__()
        self.latent_correlation_layer = LatentCorrelationLayer(num_turbines, embed_size)
        self.gcn_modules = nn.ModuleList([GCNLayer(embed_size, gcn_out_channels) for _ in range(num_turbines)])
        self.informer = InformerModule(gcn_out_channels * num_turbines, informer_out_dim)
        self.fc = nn.Linear(informer_out_dim, 1)

    def forward(self, x, edge_index_list):
        # x: [num_turbines, time_steps, embed_size]
        # 1. 计算潜在相关性
        correlated_x, attention_weights = self.latent_correlation_layer(x)

        # 2. 使用GCN提取特征
        gcn_outputs = []
        for i in range(len(self.gcn_modules)):
            turbine_x = correlated_x[i]  # 选择每个涡轮机的数据
            turbine_x = turbine_x.view(-1, turbine_x.size(-1))  # 展开为适应GCN输入格式
            gcn_out = self.gcn_modules[i](turbine_x, edge_index_list[i])
            gcn_outputs.append(gcn_out)

        # 将每个涡轮机的特征拼接
        gcn_outputs = torch.stack(gcn_outputs, dim=0).view(x.size(0),
                                                           -1)  # [num_turbines, gcn_out_channels * num_turbines]

        # 3. 使用Informer网络进行时间序列预测
        informer_out = self.informer(gcn_outputs.unsqueeze(0))  # 添加batch维度

        # 4. 最后全连接层获取最终输出
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
dummy_edge_index_list = [torch.randint(0, num_turbines, (2, 30)) for _ in range(num_turbines)]  # 每个风力涡轮机的图连接关系

output, attention_weights = model(dummy_x, dummy_edge_index_list)
print("Model output:", output)
print("Attention weights:", attention_weights)
