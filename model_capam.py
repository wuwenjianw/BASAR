"""
HGAMLP模型实现 - 异构图注意力多层感知机

该文件包含了HGAMLP(Heterogeneous Graph Attention Multi-Layer Perceptron)模型的核心实现。
主要组件包括:
1. Transformer: 自注意力机制，用于特征聚合
2. Conv1d1x1: 1D卷积层，用于跨通道特征变换  
3. FeedForwardNet: 前馈网络，用于特征投影
4. GlobalMetaAggregator: 核心模型，实现多尺度元路径特征聚合

版本说明: 此版本加入了embedding对齐机制
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch_sparse import SparseTensor
import numpy as np 


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.sum(dim=2).size()  # 为自注意力机制生成掩码，处理填充（PAD）标记。
    batch_size, len_k = seq_k.sum(dim=2).size()  # [batch_size, len_q(51), in_dim(15)]
    # eq(zero) is PAD token 检查每个元素是否为填充标记
    pad_attn_mask_k = seq_q.eq(0).all(2).data.eq(1).unsqueeze(1)  # batch_size x 1 x len_q, one is masking
    pad_attn_mask_q = seq_k.eq(0).all(2).data.eq(1).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    pad_attn_mask_k = pad_attn_mask_k.expand(batch_size, len_k, len_q).permute(0, 2, 1)
    pad_attn_mask_q = pad_attn_mask_q.expand(batch_size, len_q, len_k)
    return ~torch.logical_and(~pad_attn_mask_k, ~pad_attn_mask_q)  # batch_size x len_q x len_k 表示哪些位置需要被mask以避免干扰计算

def weighted_harmonic_similarity(data, binary_weight=0.5, distance_weight=0.5):
    """
    带权重的调和平均法节点相似性计算
    
    Args:
        data: torch.Tensor, shape [batch, graph_size, 7]
            前5维为二进制特征，后2维为距离特征
        binary_weight: float, 二进制特征的权重
        distance_weight: float, 距离特征的权重
    
    Returns:
        similarity_matrix: torch.Tensor, shape [batch, graph_size, graph_size]
    """
    
    # 分离特征
    binary_features = data[:, :, :5]  # [batch, graph_size, 5]
    distance_features = data[:, :, 5:]  # [batch, graph_size, 2]
    
    # 计算余弦相似度 (针对二进制特征)
    binary_i = binary_features.unsqueeze(2)  # [batch, graph_size, 1, 5]
    binary_j = binary_features.unsqueeze(1)  # [batch, 1, graph_size, 5]
    dot_product = torch.sum(binary_i * binary_j, dim=-1)
    norm_i = torch.norm(binary_features, dim=-1, keepdim=True)
    norm_j = torch.norm(binary_features, dim=-1, keepdim=True)
    norm_product = norm_i * norm_j.transpose(-1, -2)
    cosine_sim = dot_product / (norm_product + 1e-8)
    
    # 计算欧几里得相似性 (针对距离特征)
    dist_i = distance_features.unsqueeze(2)  # [batch, graph_size, 1, 2]
    dist_j = distance_features.unsqueeze(1)  # [batch, 1, graph_size, 2]
    euclidean_dist = torch.norm(dist_i - dist_j, dim=-1)
    euclidean_sim = torch.exp(-euclidean_dist)  # 使用指数衰减转换为相似性
    
    # 加权调和平均
    weighted_cosine = binary_weight * cosine_sim
    weighted_euclidean = distance_weight * euclidean_sim
    
    denominator = weighted_cosine + weighted_euclidean + 1e-8
    harmonic_similarity = (2.0 * weighted_cosine * weighted_euclidean) / denominator
    
    return harmonic_similarity

def build_adjacency_and_degree_matrices(data, binary_weight=0.5, distance_weight=0.5):
    """
    基于多特征相似性构建邻接矩阵和度矩阵
    
    Args:
        data: torch.Tensor, shape [batch, graph_size, 7]
            前5维为二进制特征，后2维为距离特征
        binary_weight: float, 二进制特征权重
        distance_weight: float, 距离特征权重
    
    Returns:
        A: torch.Tensor, shape [batch, graph_size, graph_size] - 邻接矩阵
        D: torch.Tensor, shape [batch, graph_size, graph_size] - 度矩阵
    """
    num_samples, num_locations, _ = data.size()
    device = data.device
    eps = 1e-8
    
    # 使用调和平均法计算相似性矩阵
    similarity_matrix = weighted_harmonic_similarity(data, binary_weight, distance_weight)
    
    # 创建对角线掩码，用于将对角线元素设为0（节点不与自己连接）
    eye_mask = torch.eye(num_locations, device=device).expand(
        (num_samples, num_locations, num_locations)).bool()
    
    # 构建邻接矩阵 A
    A = similarity_matrix.clone()
    A = A.masked_fill(eye_mask, 0)  # 对角线设为0
    
    # ===== 数值稳定性处理 =====
    # 处理NaN值
    A[A != A] = 0
    
    # 限制邻接矩阵的值域，避免数值爆炸
    A = torch.clamp(A, min=0, max=100)
    
    # 归一化处理
    A_max = A.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    A = A / (A_max + eps)
    
    # 计算度矩阵 D
    degree = A.sum(-1).clamp(min=eps)  # 避免度数为0
    D = torch.diag_embed(degree)
    
    return A, D


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp

        h_flat = h.reshape(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.reshape(-1, input_dim)  # (batch_size*n_query)*input_dim

        shape_k = (batch_size, target_size, -1)
        shape_q = (batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # batch_size*targets_size*key_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))  # batch_size*n_query*targets_size
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            mask = mask.view(batch_size, -1, target_size).expand_as(U)  # copy for n_heads times
            U[mask.bool()] = -1e8
        attention = torch.softmax(U, dim=-1)  # batch_size*n_query*targets_size
        logp_list = torch.log_softmax(U, dim=-1)  # batch_size*n_query*targets_size

        probs = attention  # probs: [batch_size, n_query, targets_size]

        return probs, logp_list


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)  # 归一化因子，用于缩放注意力分数

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)  # 常用的初始化方法

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()  # target_size 是h的大小（即图的大小）
        n_query = q.size(1)  # n_query = target_size in tsp

        h_flat = h.contiguous().view(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.contiguous().view(-1, input_dim)  # (batch_size*n_query)*input_dim
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(h_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # 计算注意力分数U n_heads*batch_size*n_query*targets_size

        if mask is not None:
            mask = mask.view(1, batch_size, -1, target_size).expand_as(U)  # copy for n_heads times
            # U[mask.bool()] = -np.inf
            U[mask.bool()] = -np.inf
        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            attnc = attention.clone()
            attnc[mask.bool()] = 0
            attention = attnc
        # print(attention)

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(batch_size, n_query, self.embedding_dim)

        return out  # batch_size*n_query*embedding_dim


class GateFFNDense(nn.Module):
    def __init__(self, model_dim, hidden_unit=512):
        super(GateFFNDense, self).__init__()
        self.W = nn.Linear(model_dim, hidden_unit, bias=False)
        self.V = nn.Linear(model_dim, hidden_unit, bias=False)
        self.W2 = nn.Linear(hidden_unit, model_dim, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, hidden_states):
        hidden_act = self.act(self.W(hidden_states))
        hidden_linear = self.V(hidden_states)
        hidden_states = hidden_act * hidden_linear
        hidden_states = self.W2(hidden_states)
        return hidden_states


class GateFFNLayer(nn.Module):
    def __init__(self, model_dim):
        super(GateFFNLayer, self).__init__()
        self.DenseReluDense = GateFFNDense(model_dim)
        self.layer_norm = Normalization(model_dim)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        return forwarded_states


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = GateFFNLayer(embedding_dim)

    def forward(self, src, mask=None):
        h0 = src
        h = self.normalization1(src)
        h = self.multiHeadAttention(q=h, mask=mask)
        h = h + h0
        h1 = h
        h = self.feedForward(h)
        h = h + h1
        return h


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(embedding_dim, n_head)
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.feedForward = GateFFNLayer(embedding_dim)
        self.normalization1 = Normalization(embedding_dim)
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, dec_self_attn_mask, dec_enc_attn_mask):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization2(memory)
        h = self.multiHeadAttention(q=tgt, h=memory, mask=dec_enc_attn_mask)
        h = h + h0
        h1 = h
        h = self.feedForward(h)
        h = h + h1
        return h

class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=4, n_layer=2):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src

class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=4, n_layer=2):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, dec_self_attn_mask=None, dec_enc_attn_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, dec_self_attn_mask, dec_enc_attn_mask)
        return tgt

class Transformer(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none'):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.query.weight, gain=gain)
        xavier_uniform_(self.key.weight, gain=gain)
        xavier_uniform_(self.value.weight, gain=gain)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, x, mask=None):
        B, M, C = x.size() # batchsize, num_metapaths, channels
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))

        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1) # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h) # [B, H, M, -1]
        return o.permute(0,2,1,3).reshape((B, M, C)) + x
    
class Conv1d1x1(nn.Module):
    """1x1一维卷积层
    
    实现了可分组的1x1卷积操作，用于跨通道的特征变换。
    支持不同的通道格式和分组卷积模式。
    """
    def __init__(self, cin, cout, groups, bias=True, cformat='channel-first'):
        """
        参数:
            cin: 输入通道数
            cout: 输出通道数  
            groups: 分组数，控制卷积的连接模式
            bias: 是否使用偏置项
            cformat: 通道格式，'channel-first'或'channel-last'
        """
        super(Conv1d1x1, self).__init__()
        self.cin = cin           # 输入通道数
        self.cout = cout         # 输出通道数
        self.groups = groups     # 分组数
        self.cformat = cformat   # 通道格式
        
        # 如果不使用偏置，设置为None
        if not bias:
            self.bias = None
            
        if self.groups == 1:  # 全连接模式：所有输入通道共享同一个卷积核
            # 权重矩阵：[cin, cout]
            self.W = nn.Parameter(torch.randn(self.cin, self.cout))
            if bias:
                # 偏置向量：[1, cout]，便于广播
                self.bias = nn.Parameter(torch.zeros(1, self.cout))
        else:  # 分组模式：每个组有独立的卷积核
            # 权重矩阵：[groups, cin, cout]，每组一个独立的变换矩阵
            self.W = nn.Parameter(torch.randn(self.groups, self.cin, self.cout))
            if bias:
                # 偏置矩阵：[groups, cout]，每组一个独立的偏置
                self.bias = nn.Parameter(torch.zeros(self.groups, self.cout))

    def reset_parameters(self):
        """重置网络参数
        
        使用Xavier均匀初始化权重，考虑ReLU激活函数的特性
        """
        def xavier_uniform_(tensor, gain=1.):
            """自定义Xavier均匀初始化函数"""
            fan_in, fan_out = tensor.size()[-2:]  # 获取输入和输出维度
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))  # 计算标准差
            a = math.sqrt(3.0) * std  # 计算均匀分布的边界
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")  # 计算ReLU的增益系数
        xavier_uniform_(self.W, gain=gain)     # 初始化权重
        if self.bias is not None:
            nn.init.zeros_(self.bias)          # 偏置初始化为0

    def forward(self, x):
        """前向传播
        
        根据分组模式和通道格式执行不同的卷积操作
        
        参数:
            x: 输入张量
               - channel-first格式: [batch, channels, features]
               - channel-last格式: [batch, features, channels]
               
        返回:
            输出张量，格式与输入保持一致
        """
        if self.groups == 1:  # 全连接模式
            if self.cformat == 'channel-first':
                # 输入: [batch, cin, features] -> 输出: [batch, cout, features]
                # 使用爱因斯坦求和约定进行矩阵乘法: bcm,mn->bcn
                return torch.einsum('bcm,mn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                # 输入: [batch, features, cin] -> 输出: [batch, features, cout]  
                # 使用爱因斯坦求和约定进行矩阵乘法: bmc,mn->bnc
                return torch.einsum('bmc,mn->bnc', x, self.W) + self.bias.T
            else:
                assert False, f"不支持的通道格式: {self.cformat}"
        else:  # 分组模式
            if self.cformat == 'channel-first':
                # 输入: [batch, groups*cin, features] -> 输出: [batch, groups*cout, features]
                # 分组卷积: bcm,cmn->bcn (c维度对应groups)
                return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                # 输入: [batch, features, groups*cin] -> 输出: [batch, features, groups*cout]
                # 分组卷积: bmc,cmn->bnc (c维度对应groups)
                return torch.einsum('bmc,cmn->bnc', x, self.W) + self.bias.T
            else:
                assert False, f"不支持的通道格式: {self.cformat}"

class FeedForwardNet(nn.Module):
    """前馈神经网络
    实现多层全连接神经网络，用于特征变换和分类输出。
    支持可变层数、激活函数和dropout正则化。
    """
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        """
        参数:
            in_feats: 输入特征维度
            hidden: 隐藏层维度  
            out_feats: 输出特征维度
            n_layers: 网络层数
            dropout: dropout概率
        """
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()  # 存储所有线性层的模块列表
        self.n_layers = n_layers       # 网络层数
        
        # 根据层数构建网络结构
        if n_layers == 1:
            # 单层网络：直接从输入映射到输出
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            # 多层网络：输入层 + 隐藏层 + 输出层
            self.layers.append(nn.Linear(in_feats, hidden))    # 输入层到第一个隐藏层
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))   # 中间隐藏层
            self.layers.append(nn.Linear(hidden, out_feats))   # 最后隐藏层到输出层
        
        # 只有多层网络才需要激活函数和dropout
        if self.n_layers > 1:
            self.prelu = nn.PReLU()        # 参数化ReLU激活函数
            self.dropout = nn.Dropout(dropout)  # Dropout正则化层
        
        self.reset_parameters()  # 初始化网络参数

    def reset_parameters(self):
        """重置网络参数
        
        使用Xavier均匀初始化权重，偏置初始化为0
        """
        gain = nn.init.calculate_gain("relu")  # 计算ReLU激活函数的增益
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)  # Xavier均匀初始化权重
            nn.init.zeros_(layer.bias)                        # 偏置初始化为0

    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入特征张量 [batch_size, in_feats]
            
        返回:
            输出特征张量 [batch_size, out_feats]
        """
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)  # 线性变换
            # 除了最后一层，其他层都需要激活函数和dropout
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))  # 激活函数 + dropout
        return x


class CapamMetaAggregator(nn.Module):
    def __init__(self, agent_input_dim, task_input_dim, embedding_dim, device):
        """
        参数:
            agent_input_dim: 智能体输入特征维度
            task_input_dim: 任务输入特征维度
            embedding_dim: 嵌入维度
            device: 计算设备
        """

        super(CapamMetaAggregator, self).__init__()
        self.in_feats = embedding_dim
        self.device = device
        hidden = embedding_dim

        # ========================new===============
        self.agent_embedding = nn.Linear(agent_input_dim, embedding_dim)  #! 统一嵌入到128维
        self.task_embedding = nn.Linear(task_input_dim, embedding_dim)
        self.agentEncoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.crossDecoder1 = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
        self.crossDecoder2 = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
        self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)
        self.globalDecoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
        self.pointer = SingleHeadAttention(embedding_dim)
        self.taskEncoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        
        # CapAM的参数
        # 用于处理图的不同变换
        self.init_embed = nn.Linear(128, 128 * 3)  # 将节点维度从100变为128*3
        self.W_L_1_G1 = nn.Linear(128 * 3 * 3, 128)
        self.W_L_1_G2 = nn.Linear(128 * 3 * 3, 128)
        self.W_L_1_G3 = nn.Linear(128 * 3 * 3, 128)
        self.fusion_2 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.activ = nn.LeakyReLU()  # 使用 LeakyReLU 激活函数
        self.W_F = nn.Linear(128 * 3, 128)  # 最后一层全连接层
        self.normalization_capam = Normalization(128 * 3)  # 使用BatchNorm1d归一化


    def encoding_tasks(self, task_embedding, agent_embedding, tasks, agents, mask, global_mask, decis_index):
        # =====================================================================
        # !10.29 使用剩余需求和距离进行嵌入
        X_binary = tasks[:, :, :5]              # [B, N, 5]
        X_loc = tasks[:, :, -3:-1]                # [B, N, 2]
        # 将位置特征和二进制特征合并
        X_combined = torch.cat([X_binary, X_loc], dim=-1)

        # 构建邻接矩阵和度矩阵
        A, D = build_adjacency_and_degree_matrices(X_combined, binary_weight=0.3, distance_weight=0.7)

        # ===== 修复4: 初始特征嵌入并检查NaN =====
        F0 = self.init_embed(task_embedding)
        
        # 检查F0是否包含NaN
        if torch.isnan(F0).any():
            print("Warning: NaN detected in F0, replacing with zeros")
            F0 = torch.where(torch.isnan(F0), torch.zeros_like(F0), F0)
        
        F0_squared = torch.mul(F0[:, :, :], F0[:, :, :])  # 计算平方
        F0_cube = torch.mul(F0[:, :, :], F0_squared[:, :, :])  # 计算立方
        
        # 计算图的拉普拉斯矩阵 L  K = 3
        L = D - A
        
        # ===== 修复5: 使用更稳定的矩阵乘法 =====
        # 限制拉普拉斯矩阵的范数，避免数值爆炸
        L_norm = torch.norm(L, dim=(-2, -1), keepdim=True).clamp(min=1.0)
        L = L / L_norm  # 归一化拉普拉斯矩阵
        
        L_squared = torch.matmul(L, L)  # 计算拉普拉斯矩阵的平方

        # 图卷积操作
        LF0 = torch.matmul(L, F0)
        L2F0 = torch.matmul(L_squared, F0)
        LF0_sq = torch.matmul(L, F0_squared)
        L2F0_sq = torch.matmul(L_squared, F0_squared)
        LF0_cube = torch.matmul(L, F0_cube)
        L2F0_cube = torch.matmul(L_squared, F0_cube)

        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :], LF0[:, :, :], L2F0[:, :, :]), -1))
        g_L1_2 = self.W_L_1_G2(torch.cat((F0_squared[:, :, :], LF0_sq[:, :, :], L2F0_sq[:, :, :]), -1))
        g_L1_3 = self.W_L_1_G3(torch.cat((F0_cube[:, :, :], LF0_cube[:, :, :], L2F0_cube[:, :, :]), -1))
        
        # ===== 修复6: 检查中间结果 =====
        if torch.isnan(g_L1_1).any() or torch.isnan(g_L1_2).any() or torch.isnan(g_L1_3).any():
            print("Warning: NaN detected in g_L1_x")
            print(f"g_L1_1 NaN: {torch.isnan(g_L1_1).any()}")
            print(f"g_L1_2 NaN: {torch.isnan(g_L1_2).any()}")
            print(f"g_L1_3 NaN: {torch.isnan(g_L1_3).any()}")
            # 替换NaN为0
            g_L1_1 = torch.where(torch.isnan(g_L1_1), torch.zeros_like(g_L1_1), g_L1_1)
            g_L1_2 = torch.where(torch.isnan(g_L1_2), torch.zeros_like(g_L1_2), g_L1_2)
            g_L1_3 = torch.where(torch.isnan(g_L1_3), torch.zeros_like(g_L1_3), g_L1_3)

        # 合并和激活
        F1 = torch.cat((g_L1_1, g_L1_2, g_L1_3), -1)
        
        # ===== 修复7: 先归一化再激活，调整顺序 =====
        F1 = self.normalization_capam(F1)  # 先归一化
        F1 = self.activ(F1 + F0)  # 再激活和残差连接

        # ===== 修复8: 检查F1是否包含NaN =====
        if torch.isnan(F1).any():
            print("Warning: NaN detected in F1 after normalization")
            print(f"NaN positions: {torch.isnan(F1).sum()} out of {F1.numel()}")
            F1 = torch.where(torch.isnan(F1), torch.zeros_like(F1), F1)

        # 最终嵌入【2，100，128】
        task_encoding = self.activ(self.W_F(F1))
        
        # ===== 修复9: 最后检查task_encoding =====
        if torch.isnan(task_encoding).any():
            print("Warning: NaN in final task_encoding, using fallback")
            task_encoding = torch.where(torch.isnan(task_encoding), 
                                        torch.zeros_like(task_encoding), 
                                        task_encoding)
        
        # 加入注意力机制
        # task_encoding = self.taskEncoder(task_encoding, mask)  # task_encoding

        embedding_dim = task_encoding.size(-1)
        mean_mask = mask[:,0,:].unsqueeze(2).repeat(1, 1, embedding_dim)  # mean_mask: [batch_size, n_task(51), 128]
        
        # ===== 修复10: 使用更稳定的聚合方式 =====
        # 使用0代替nan，避免全NaN情况
        compressed_task = torch.where(mean_mask, torch.tensor(0.0, device=task_encoding.device), task_embedding)
        
        # 使用sum+count代替nanmean，更稳定
        valid_counts = (~mean_mask[:,:,0]).sum(dim=1, keepdim=True).clamp(min=1)
        aggregated_tasks = compressed_task.sum(dim=1, keepdim=True) / valid_counts.unsqueeze(-1)  # aggregated_tasks: [batch_size, 1, 128]

        return aggregated_tasks, task_encoding


    def encoding_agents(self, agents_embedding, mask=None):
        agents_encoding = self.agentEncoder(agents_embedding, mask)  # agents_encoding: [batch_size, n_agent(15), 128]
        embedding_dim = agents_encoding.size(-1) 
        mean_mask = mask[:,0,:].unsqueeze(2).repeat(1, 1, embedding_dim)  # mean_mask: [batch_size, n_agent(15), 128]
        compressed_task = torch.where(mean_mask, torch.nan, agents_embedding)  
        aggregated_agents = torch.nanmean(compressed_task, dim=1).unsqueeze(1)  # aggregated_agents: [batch_size, 1, 128]
        return aggregated_agents, agents_encoding

    def forward(self, tasks, agents, global_mask, index):

        # mask掩码
        task_mask = get_attn_pad_mask(tasks, tasks)  # task: [batch_size, len_q(51), in_dim(15)]; task_mask: [batch_size, len_q(51), len_q(51)]
        agent_mask = get_attn_pad_mask(agents, agents)  # agents: [batch_size, len_q(15), in_dim(11)]; angent_mask: [batch_size, len_q(15), len_q(15)]
        agent_task_mask = get_attn_pad_mask(agents, tasks)  # agent_task_mask: [batch_size, len_q(15), len_q(51)]
        
        task_embedding = self.task_embedding(tasks)
        agent_embedding = self.agent_embedding(agents)

        aggregated_agents, agents_encoding = self.encoding_agents(agent_embedding, mask=agent_mask)  # aggregated_agents: [batch_size, 1, 128]
        aggregated_task, task_encoding = self.encoding_tasks(task_embedding, agent_embedding, tasks, agents, mask=task_mask, global_mask=global_mask,decis_index=index)
        
        # agent_task_feature = self.crossDecoder2(agents_encoding, task_encoding, None, agent_task_mask)  # agent_task_feature: [batch_size, n_agent(15), 128]
        # current_state1 = torch.gather(agent_task_feature, 1, index.repeat(1, 1, agent_task_feature.size(2)))  # current_state1:[batch_size, 1, 128]  从每个agent的特征中选择元素, index变为: [batch_size, 1, 128]
        # current_state = self.fusion(torch.cat((current_state1, aggregated_task, aggregated_agents), dim=-1))  # current_state:[batch_size, 1, 128]
        # current_state_prime = self.globalDecoder(current_state, task_encoding, None, global_mask)  # current_state_prime: [batch_size, 1, 128]
        # probs, logps = self.pointer(current_state_prime, task_encoding, mask=global_mask)
        # logps = logps.squeeze(1)  # 去除数组中所有维度为 1 的轴
        # probs = probs.squeeze(1)

        # ====================== 10.27添加 ============================
        task_agent_mask = get_attn_pad_mask(tasks, agents) 
        task_agent_feature = self.crossDecoder1(task_encoding, agents_encoding, None, task_agent_mask)  # task_agent_feature: [batch_size, n_task(51), 128]
        agent_task_feature = self.crossDecoder2(agents_encoding, task_encoding, None, agent_task_mask)  # agent_task_feature: [batch_size, n_agent(15), 128]
        current_state1 = torch.gather(agent_task_feature, 1, index.repeat(1, 1, agent_task_feature.size(2)))  # current_state1:[batch_size, 1, 128]  从每个agent的特征中选择元素, index变为: [batch_size, 1, 128]
        current_state = self.fusion_2(torch.cat((current_state1, aggregated_agents), dim=-1))  # current_state:[batch_size, 1, 128]
        current_state_prime = self.globalDecoder(current_state, task_agent_feature, None, global_mask)  # current_state_prime: [batch_size, 1, 128]
        probs, logps = self.pointer(current_state_prime, task_agent_feature, mask=global_mask)
        logps = logps.squeeze(1)  # 去除数组中所有维度为 1 的轴
        probs = probs.squeeze(1)

        return probs, logps 


# from worker import *

# if "__main__":
#     device='cpu'

#     worker = Worker(1, None, None, 0, device=device, seed=1, save_image=False)
#     release_agents, current_time = worker.env.next_decision()
#     # 随机打乱智能体顺序，增加探索性
#     random.shuffle(release_agents[0])
#     finished_task = []
#     agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
#     agent = worker.env.agent_dic[agent_id]
#     task_info, total_agents, mask = worker.convert_torch(worker.env.agent_observe(agent_id, False))

#     model = GlobalMetaAggregator(['A', 'AA', 'AP', 'AAA', 'AAP', 'APA'], 11, 15, 128, [0.], 0.1, 0.1, device)

#     model(task_info, total_agents, mask, torch.tensor([[[agent_id]]]))