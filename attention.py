import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp.autocast_mode import autocast
from parameters import *


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.sum(dim=2).size()  # 为自注意力机制生成掩码，处理填充（PAD）标记。
    batch_size, len_k = seq_k.sum(dim=2).size()  # [batch_size, len_q(51), in_dim(15)]
    # eq(zero) is PAD token 检查每个元素是否为填充标记
    pad_attn_mask_k = seq_q.eq(0).all(2).data.eq(1).unsqueeze(1)  # batch_size x 1 x len_q, one is masking
    pad_attn_mask_q = seq_k.eq(0).all(2).data.eq(1).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    pad_attn_mask_k = pad_attn_mask_k.expand(batch_size, len_k, len_q).permute(0, 2, 1)
    pad_attn_mask_q = pad_attn_mask_q.expand(batch_size, len_q, len_k)
    return ~torch.logical_and(~pad_attn_mask_k, ~pad_attn_mask_q)  # batch_size x len_q x len_k 表示哪些位置需要被mask以避免干扰计算


def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.logical_not(np.triu(np.ones(attn_shape), k=0)).astype(int)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


def masked_token_mean(sequence, mask=None):
    """
    对 token 序列做带 mask 的均值池化。

    Args:
        sequence: [batch_size, seq_len, embedding_dim]
        mask: [batch_size, 1, seq_len] 或 [batch_size, seq_len]，True 表示被屏蔽
    """
    if mask is None:
        return sequence.mean(dim=1, keepdim=True)

    if mask.dim() == 3:
        token_mask = mask[:, 0, :]
    else:
        token_mask = mask
    token_mask = token_mask.bool()

    valid = (~token_mask).unsqueeze(-1).to(sequence.dtype)
    denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    pooled = (sequence * valid).sum(dim=1, keepdim=True) / denom
    return pooled


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


class GlobalStateMLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=None):
        super(GlobalStateMLP, self).__init__()
        hidden_dim = hidden_dim or embedding_dim * 2
        self.input_norm = nn.LayerNorm(embedding_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, current_state, pooled_memory):
        fused = torch.cat((current_state, pooled_memory), dim=-1)
        fused = self.input_norm(fused)
        return current_state + self.mlp(fused)


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


class AttentionNet(nn.Module):
    def __init__(self, agent_input_dim, task_input_dim, embedding_dim,
                 cross_attention_mode=None, global_decoder_mode=None):
        super(AttentionNet, self).__init__()
        self.cross_attention_mode = (cross_attention_mode or getattr(TrainParams, 'CROSS_ATTENTION_MODE', 'dual_cross')).lower()
        self.global_decoder_mode = (global_decoder_mode or getattr(TrainParams, 'GLOBAL_DECODER_MODE', 'attention')).lower()
        if self.cross_attention_mode not in {'dual_cross', 'shared_self'}:
            raise ValueError(f'Unsupported cross_attention_mode: {self.cross_attention_mode}')
        if self.global_decoder_mode not in {'attention', 'mlp'}:
            raise ValueError(f'Unsupported global_decoder_mode: {self.global_decoder_mode}')

        self.agent_embedding = nn.Linear(agent_input_dim, embedding_dim)  #! 统一嵌入到128维
        self.task_embedding = nn.Linear(task_input_dim, embedding_dim)  # layer for input information
        self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)

        self.taskEncoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.agentEncoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        if self.cross_attention_mode == 'dual_cross':
            self.crossDecoder1 = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
            self.crossDecoder2 = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
            self.sharedSelfAttention = None
        else:
            # 消融 1：用单一共享的自注意力模块替代 task-agent / agent-task 双交叉注意力
            self.crossDecoder1 = None
            self.crossDecoder2 = None
            self.sharedSelfAttention = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
        if self.global_decoder_mode == 'attention':
            self.globalDecoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
            self.globalStateMLP = None
        else:
            # 消融 2：用 MLP 近似当前决策状态提取，不再对任务 token 做 cross attention
            self.globalDecoder = None
            self.globalStateMLP = GlobalStateMLP(
                embedding_dim=embedding_dim,
                hidden_dim=getattr(TrainParams, 'GLOBAL_MLP_HIDDEN_DIM', embedding_dim * 2),
            )
        self.pointer = SingleHeadAttention(embedding_dim)
        # self.LSTM = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)

    def encoding_tasks(self, task_inputs, mask=None):
        task_embedding = self.task_embedding(task_inputs)  # task_inputs: [batch_size, n_task(51), 128]
        task_encoding = self.taskEncoder(task_embedding, mask)  # task_encoding: [batch_size, n_task(51), 128]
        embedding_dim = task_encoding.size(-1)
        mean_mask = mask[:,0,:].unsqueeze(2).repeat(1, 1, embedding_dim)  # mean_mask: [batch_size, n_task(51), 128]
        compressed_task = torch.where(mean_mask, torch.nan, task_embedding)  # 对无效的任务输入进行标记
        aggregated_tasks = torch.nanmean(compressed_task, dim=1).unsqueeze(1)  # aggregated_tasks: [batch_size, 1, 128]
        return aggregated_tasks, task_encoding

    def encoding_agents(self, agents_inputs, mask=None):
        agents_embedding = self.agent_embedding(agents_inputs)  # agents_embedding: [batch_size, n_agent(15), 128]
        agents_encoding = self.agentEncoder(agents_embedding, mask)  # agents_encoding: [batch_size, n_agent(15), 128]
        embedding_dim = agents_encoding.size(-1) 
        mean_mask = mask[:,0,:].unsqueeze(2).repeat(1, 1, embedding_dim)  # mean_mask: [batch_size, n_agent(15), 128]
        compressed_task = torch.where(mean_mask, torch.nan, agents_embedding)  
        aggregated_agents = torch.nanmean(compressed_task, dim=1).unsqueeze(1)  # aggregated_agents: [batch_size, 1, 128]
        return aggregated_agents, agents_encoding

    def _build_interaction_features(
        self,
        task_encoding,
        agents_encoding,
        task_mask,
        agent_mask,
        task_agent_mask,
        agent_task_mask,
    ):
        if self.cross_attention_mode == 'dual_cross':
            task_agent_feature = self.crossDecoder1(task_encoding, agents_encoding, None, task_agent_mask)
            agent_task_feature = self.crossDecoder2(agents_encoding, task_encoding, None, agent_task_mask)
        else:
            task_agent_feature = self.sharedSelfAttention(task_encoding, task_mask)
            agent_task_feature = self.sharedSelfAttention(agents_encoding, agent_mask)
        return task_agent_feature, agent_task_feature

    def _refine_current_state(self, current_state, task_agent_feature, global_mask):
        if self.global_decoder_mode == 'attention':
            return self.globalDecoder(current_state, task_agent_feature, None, global_mask)
        pooled_task_context = masked_token_mean(task_agent_feature, global_mask)
        return self.globalStateMLP(current_state, pooled_task_context)

    def forward(self, tasks, agents, global_mask, index):
        task_mask = get_attn_pad_mask(tasks, tasks)  # task: [batch_size, len_q(51), in_dim(15)]; task_mask: [batch_size, len_q(51), len_q(51)]
        agent_mask = get_attn_pad_mask(agents, agents)  # agents: [batch_size, len_q(15), in_dim(11)]; angent_mask: [batch_size, len_q(15), len_q(15)]
        task_agent_mask = get_attn_pad_mask(tasks, agents)  # task_agent_mask: [batch_size, len_q(51), len_q(15)]
        agent_task_mask = get_attn_pad_mask(agents, tasks)  # agent_task_mask: [batch_size, len_q(15), len_q(51)]
        aggregated_task, task_encoding = self.encoding_tasks(tasks, mask=task_mask)  #  task_encoding: [batch_size, n_task(51), 128]
        aggregated_agents, agents_encoding = self.encoding_agents(agents, mask=agent_mask)  # aggregated_agents: [batch_size, 1, 128]
        task_agent_feature, agent_task_feature = self._build_interaction_features(
            task_encoding,
            agents_encoding,
            task_mask,
            agent_mask,
            task_agent_mask,
            agent_task_mask,
        )
        current_state1 = torch.gather(agent_task_feature, 1, index.repeat(1, 1, agent_task_feature.size(2)))  # current_state1:[batch_size, 1, 128]  从每个agent的特征中选择元素, index变为: [batch_size, 1, 128]
        current_state = self.fusion(torch.cat((current_state1, aggregated_task, aggregated_agents), dim=-1))  # current_state:[batch_size, 1, 128]
        current_state_prime = self._refine_current_state(current_state, task_agent_feature, global_mask)
        probs, logps = self.pointer(current_state_prime, task_agent_feature, mask=global_mask)
        logps = logps.squeeze(1)  # 去除数组中所有维度为 1 的轴
        probs = probs.squeeze(1)
        return probs, logps  # probs: [batch_size, n_task(51)], logps: [batch_size, n_task(51)]


class ImproMetaNet(nn.Module):
    def __init__(self, agent_input_dim, task_input_dim, embedding_dim,
                 cross_attention_mode=None, global_decoder_mode=None):
        super(ImproMetaNet, self).__init__()
        self.cross_attention_mode = (cross_attention_mode or getattr(TrainParams, 'CROSS_ATTENTION_MODE', 'dual_cross')).lower()
        self.global_decoder_mode = (global_decoder_mode or getattr(TrainParams, 'GLOBAL_DECODER_MODE', 'attention')).lower()
        if self.cross_attention_mode not in {'dual_cross', 'shared_self'}:
            raise ValueError(f'Unsupported cross_attention_mode: {self.cross_attention_mode}')
        if self.global_decoder_mode not in {'attention', 'mlp'}:
            raise ValueError(f'Unsupported global_decoder_mode: {self.global_decoder_mode}')

        self.agent_embedding = nn.Linear(agent_input_dim, embedding_dim)  #! 统一嵌入到128维
        self.task_embedding = nn.Linear(task_input_dim, embedding_dim)  # layer for input information
        self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)

        self.taskEncoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.agentEncoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        if self.cross_attention_mode == 'dual_cross':
            self.crossDecoder1 = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
            self.crossDecoder2 = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
            self.sharedSelfAttention = None
        else:
            self.crossDecoder1 = None
            self.crossDecoder2 = None
            self.sharedSelfAttention = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
        if self.global_decoder_mode == 'attention':
            self.globalDecoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
            self.globalStateMLP = None
        else:
            self.globalDecoder = None
            self.globalStateMLP = GlobalStateMLP(
                embedding_dim=embedding_dim,
                hidden_dim=getattr(TrainParams, 'GLOBAL_MLP_HIDDEN_DIM', embedding_dim * 2),
            )
        self.pointer = SingleHeadAttention(embedding_dim)
        self.output_bottleneck_dim = int(getattr(TrainParams, 'IMPRO_BOTTLENECK_DIM', 16))
        self.output_down = nn.Linear(embedding_dim, self.output_bottleneck_dim)
        self.output_up = nn.Linear(self.output_bottleneck_dim, embedding_dim)
        self._latest_sparse_embedding = None
        # self.LSTM = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)

    def encoding_tasks(self, task_inputs, mask=None):
        task_embedding = self.task_embedding(task_inputs)  # task_inputs: [batch_size, n_task(51), 128]
        task_encoding = self.taskEncoder(task_embedding, mask)  # task_encoding: [batch_size, n_task(51), 128]
        embedding_dim = task_encoding.size(-1)
        mean_mask = mask[:,0,:].unsqueeze(2).repeat(1, 1, embedding_dim)  # mean_mask: [batch_size, n_task(51), 128]
        compressed_task = torch.where(mean_mask, torch.nan, task_embedding)  # 对无效的任务输入进行标记
        aggregated_tasks = torch.nanmean(compressed_task, dim=1).unsqueeze(1)  # aggregated_tasks: [batch_size, 1, 128]
        return aggregated_tasks, task_encoding

    def encoding_agents(self, agents_inputs, mask=None):
        agents_embedding = self.agent_embedding(agents_inputs)  # agents_embedding: [batch_size, n_agent(15), 128]
        agents_encoding = self.agentEncoder(agents_embedding, mask)  # agents_encoding: [batch_size, n_agent(15), 128]
        embedding_dim = agents_encoding.size(-1)
        mean_mask = mask[:,0,:].unsqueeze(2).repeat(1, 1, embedding_dim)  # mean_mask: [batch_size, n_agent(15), 128]
        compressed_task = torch.where(mean_mask, torch.nan, agents_embedding)
        aggregated_agents = torch.nanmean(compressed_task, dim=1).unsqueeze(1)  # aggregated_agents: [batch_size, 1, 128]
        return aggregated_agents, agents_encoding

    def _build_interaction_features(
        self,
        task_encoding,
        agents_encoding,
        task_mask,
        agent_mask,
        task_agent_mask,
        agent_task_mask,
    ):
        if self.cross_attention_mode == 'dual_cross':
            task_agent_feature = self.crossDecoder1(task_encoding, agents_encoding, None, task_agent_mask)
            agent_task_feature = self.crossDecoder2(agents_encoding, task_encoding, None, agent_task_mask)
        else:
            task_agent_feature = self.sharedSelfAttention(task_encoding, task_mask)
            agent_task_feature = self.sharedSelfAttention(agents_encoding, agent_mask)
        return task_agent_feature, agent_task_feature

    def _refine_current_state(self, current_state, task_agent_feature, global_mask):
        if self.global_decoder_mode == 'attention':
            return self.globalDecoder(current_state, task_agent_feature, None, global_mask)
        pooled_task_context = masked_token_mean(task_agent_feature, global_mask)
        return self.globalStateMLP(current_state, pooled_task_context)

    def forward(self, tasks, agents, global_mask, index):
        task_mask = get_attn_pad_mask(tasks, tasks)  # task: [batch_size, len_q(51), in_dim(15)]; task_mask: [batch_size, len_q(51), len_q(51)]
        agent_mask = get_attn_pad_mask(agents, agents)  # agents: [batch_size, len_q(15), in_dim(11)]; angent_mask: [batch_size, len_q(15), len_q(15)]
        task_agent_mask = get_attn_pad_mask(tasks, agents)  # task_agent_mask: [batch_size, len_q(51), len_q(15)]
        agent_task_mask = get_attn_pad_mask(agents, tasks)  # agent_task_mask: [batch_size, len_q(15), len_q(51)]
        aggregated_task, task_encoding = self.encoding_tasks(tasks, mask=task_mask)  #  task_encoding: [batch_size, n_task(51), 128]
        aggregated_agents, agents_encoding = self.encoding_agents(agents, mask=agent_mask)  # aggregated_agents: [batch_size, 1, 128]
        task_agent_feature, agent_task_feature = self._build_interaction_features(
            task_encoding,
            agents_encoding,
            task_mask,
            agent_mask,
            task_agent_mask,
            agent_task_mask,
        )
        current_state1 = torch.gather(agent_task_feature, 1, index.repeat(1, 1, agent_task_feature.size(2)))  # current_state1:[batch_size, 1, 128]  从每个agent的特征中选择元素, index变为: [batch_size, 1, 128]
        current_state = self.fusion(torch.cat((current_state1, aggregated_task, aggregated_agents), dim=-1))  # current_state:[batch_size, 1, 128]
        current_state_prime = self._refine_current_state(current_state, task_agent_feature, global_mask)
        sparse_embedding = self.output_down(current_state_prime)
        self._latest_sparse_embedding = sparse_embedding
        current_state_prime = self.output_up(sparse_embedding)
        probs, logps = self.pointer(current_state_prime, task_agent_feature, mask=global_mask)
        logps = logps.squeeze(1)  # 去除数组中所有维度为 1 的轴
        probs = probs.squeeze(1)
        return probs, logps  # probs: [batch_size, n_task(51)], logps: [batch_size, n_task(51)]

    def get_l1_regularization_loss(self, l1_lambda=None):
        coeff = l1_lambda
        if coeff is None:
            coeff = float(getattr(TrainParams, 'IMPRO_L1_LAMBDA', 1e-5))
        if self._latest_sparse_embedding is None:
            device = self.output_down.weight.device
            return torch.tensor(0.0, device=device)
        return coeff * self._latest_sparse_embedding.abs().mean()


def padding_inputs(inputs):
    seq = pad_sequence(inputs, batch_first=False, padding_value=1)
    seq = seq.permute(2, 1, 0)
    mask = torch.zeros_like(seq, dtype=torch.int64)
    ones = torch.ones_like(seq, dtype=torch.int64)
    mask = torch.where(seq != 1, mask, ones)
    return seq, mask
