import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
import einops
import pywt
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import warnings
warnings.filterwarnings('ignore')


try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))
        r = torch.stack((t.real, t.imag), -1)
        return r


    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:, :, 0], x[:, :, 1]), dim=(-d))
        return t.real




# Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration, CVPR 2024.
# https://github.com/joshyZhou/AST
# https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf
class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q,k,v

# Adaptive Sparse Self-Attention (ASSA)
class WindowAttention_sparse(nn.Module):
    def __init__(self, dim, win_size, num_heads=1, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2  # b,h,w,c
        else:
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn = attn0 * w1 + attn1 * w2
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# class CNN(nn.Module):
#     def __init__(self, input_size, num_filters, kernel_size, num_layers, batch_size, device="cuda"):
#         super().__init__()
#         self.device = device
#         self.input_size = input_size
#         self.num_filters = num_filters
#         self.kernel_size = kernel_size
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#
#         layers = []
#         for i in range(num_layers):
#             in_channels = input_size if i == 0 else num_filters
#             conv_layer = nn.Conv1d(in_channels=in_channels,
#                                    out_channels=num_filters,
#                                    kernel_size=kernel_size,
#                                    padding=(kernel_size - 1) // 2)  # To maintain the same output size
#             layers.append(conv_layer)
#             layers.append(nn.ReLU())  # Activation after each conv layer
#
#         self.conv_net = nn.Sequential(*layers)
#         self.to(self.device)
#
#     def forward(self, x_enc):
#         # x_enc: (batch_size, seq_len, input_size)
#         # Permute to (batch_size, input_size, seq_len) for Conv1d
#         x_enc = x_enc.permute(0, 2, 1)
#         output = self.conv_net(x_enc)
#         # Permute back to (batch_size, seq_len, num_filters)
#         output = output.permute(0, 2, 1)
#
#         return output  # (batch_size, seq_len, num_filters)


class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cuda"):
        super(GCN,self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.layers=nn.ModuleList()
        self.batch_size=batch_size

        self.layers.append(GCNConv(input_size,hidden_size))#first layer
        for _ in range(num_layers-1):
            self.layers.append(GCNConv(hidden_size, hidden_size))#hidden layers
        self.to(self.device)

    def forward(self, x_enc, edge_index):
        for layer in self.layers:
            x_enc=F.relu(layer(x_enc,edge_index))
        return x_enc

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        # self.cnn = CNN(input_size=configs.enc_in,
        #                num_filters=configs.d_model,
        #                kernel_size=3,
        #                num_layers=2,
        #                batch_size=configs.batch_size)
        self.gcn = GCN(input_size=configs.enc_in,
                       hidden_size=configs.d_model,
                       num_layers=2,
                       batch_size=configs.batch_size)
        self.window_attention_sparse1 = WindowAttention_sparse(configs.d_model, (1, configs.pred_len))
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(configs.d_model) for l in range(configs.e_layers - 1)
            ] if configs.distil and ('forecast' in configs.task_name) else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out  # [B, L, D]

    def short_forecast(self, x_enc, edge_index, x_mark_enc, x_dec, x_mark_dec):
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        #分别读取序列数据进行处理
        # 让CNN和informer一起提取信息特征，输出两个信息矩阵，
        # 这两个信息矩阵结合一下再输入解码器里训练
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # cnn_out = self.cnn(x_enc)#初始数据经过CNN网络的特征处理
        # cnn_out = self.window_attention_sparse1(cnn_out)
        gcn_out = self.gcn(x_enc, edge_index)
        gcn_out = self.window_attention_sparse1(gcn_out)

        # CNN提取的特征经过稀疏注意力的再次处理提取输入到
        # 将CNN和Informer的输出并联进行拼接进入解码器
        # combined_features =enc_out+cnn_out
        combined_features = enc_out + gcn_out
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        combined_out, attns = self.encoder(combined_features, attn_mask=None)
        dec_out = self.decoder(dec_out, combined_out, x_mask=None, cross_mask=None)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out  # [B, L, D]



    def forward(self, x_enc, edge_index, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'short_term_forecast':
            dec_out = self.short_forecast(x_enc, edge_index, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        return None
