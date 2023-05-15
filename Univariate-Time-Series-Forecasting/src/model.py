import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim,num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

        # 使用He方法初始化线性层
        for name, param in self.fc.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.kaiming_uniform_(param)

    def forward(self, input):
        # Propagate the input trough lstm
        _, (hidden, _) = self.lstm(input)
        # Get the prediction for the next time step
        out = self.fc(hidden[-1, :, :])

        return out.view(-1, 1)


class Transformer(nn.Module):
    # d_model : number of features
    def __init__(self, feature_size, num_layers, dropout):
        super(Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=7, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, device):
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, mask)
        output = self.decoder(output)
        return output

