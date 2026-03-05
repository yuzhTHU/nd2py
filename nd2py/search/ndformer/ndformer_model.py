# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import sys
import torch
import logging
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch_geometric.utils import to_dense_batch
from typing import List, Dict, Tuple, Union, Literal
from .ndformer_config import NDformerConfig
from ... import utils

# See https://github.com/pytorch/pytorch/issues/100469
warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly.")
logger = logging.getLogger('ND2.Model')


class NDformerModel(nn.Module):
    def __init__(self, config: NDformerConfig):
        super().__init__()
        self.config = config
        
        self.data_embedder = nn.Linear(self.config.max_var_num + 1, self.config.d_emb)
        self.token_embedder = nn.Embedding(self.config.n_words, self.config.d_emb)
        
        self.gnn_encoder = utils.nn.GNN(
            self.config.d_emb, 
            self.config.n_GNN_layers, 
            self.config.d_emb,
            self.config.d_emb, 
            self.config.dropout
        )

        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=self.config.d_emb, 
            nhead=self.config.n_head, 
            dim_feedforward=self.config.d_ff, 
            dropout=self.config.dropout, 
            batch_first=True
        ), num_layers=self.config.n_transformer_encoder_layers)

        self.pe = utils.nn.PositionalEncoding(self.config.d_emb, self.config.dropout)

        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=self.config.d_emb, 
            nhead=self.config.n_head, 
            dim_feedforward=self.config.d_ff, 
            batch_first=True, 
            dropout=self.config.dropout
        ), num_layers=self.config.n_transformer_decoder_layers)

        self.fc_head = nn.Linear(self.config.d_emb, self.config.n_words) 

    def encode_graph(self, data_node, data_edge, edge_list, num_nodes, node_batch_idx, timer=None):
        """
        图编码阶段：仅在图拓扑变更或初始化时调用一次。
        """
        # Embed
        data_node = self.data_embedder(data_node.nan_to_num(0.0)) # (sample_num, batched_node_num, d_emb)
        data_edge = self.data_embedder(data_edge.nan_to_num(0.0)) # (sample_num, batched_edge_num, d_emb)
        if timer is not None:
            torch.cuda.synchronize()
            timer.add('Data-Embedding')

        # 1. 稀疏 GNN 消息传递 (只计算有效节点)
        # 输入形状假设已被拉平：(TotalNodes, InputDim)
        # 输出形状：(TotalNodes, d_emb)
        gnn_out, _ = self.gnn_encoder(data_node, data_edge, edge_list, num_nodes) # (sample_num, batched_node_num, d_emb)
        if timer is not None:
            torch.cuda.synchronize()
            timer.add('GNN-Encoder')

        # 2. 核心转换：稀疏变稠密
        # dense_nodes: (graph_num, MaxNodeNum, d_emb)
        # valid_mask: (graph_num, MaxNodeNum)，True 表示有效节点，False 表示 Pad 出来的假节点
        dense_nodes, valid_mask = to_dense_batch(
            gnn_out.transpose(0, 1), # (batched_node_num, sample_num, d_emb)
            node_batch_idx
        ) # (graph_num, maximal_node_num, sample_num, d_emb) / (graph_num, maximal_node_num, sample_num)
        if timer is not None:
            torch.cuda.synchronize()
            timer.add('To-Dense-Batch')
        # ⚠️ 【关键对齐】：PyTorch Transformer 要求的 padding_mask 中，True 代表需要被忽略的 Pad！
        # 而 PyG 返回的 mask 中，True 代表有效节点。所以这里必须按位取反 (~)。
        graph_num, node_num, sample_num, d_emb = dense_nodes.shape
        dense_nodes = dense_nodes.flatten(1, 2) # (graph_num, node_num*sample_num, d_emb)
        src_key_padding_mask = ~valid_mask[..., None].expand(graph_num, node_num, sample_num).flatten(1, 2) # (graph_num, node_num*sample_num, d_emb)

        # 3. Transformer Encoder 编码
        memory = self.transformer_encoder(
            dense_nodes, # (graph_num, node_num*sample_num, d_emb)
            src_key_padding_mask=src_key_padding_mask # (graph_num, node_num*sample_num), 其中 True 代表需要被忽略的 Pad
        ) # (graph_num, node_num*sample_num, d_emb)
        if timer is not None:
            torch.cuda.synchronize()
            timer.add('Transformer-Encoder')

        return memory, src_key_padding_mask

    def decode_sequence(self, partial_eq, memory, memory_key_padding_mask, seq2graph_idx=None, eq_pad_mask=None, timer=None):
        """
        序列解码阶段：支持 1-to-N 广播，可高频调用。
        """
        device = partial_eq.device
        b_seq, seq_len = partial_eq.size()

        # 1. 维度桥接：利用 seq2graph_idx 实现零拷贝特征广播
        if seq2graph_idx is not None:
            memory_expanded = memory[seq2graph_idx]                      # (B_seq, MaxNodeNum, d_emb)
            memory_mask_expanded = memory_key_padding_mask[seq2graph_idx] # (B_seq, MaxNodeNum)
        else:
            # 推理阶段，如果是单图输入，直接复用 memory
            memory_expanded = memory
            memory_mask_expanded = memory_key_padding_mask

        # 2. 序列嵌入与位置编码
        # seq_emb: (B_seq, seq_len, d_emb)
        seq_emb = self.token_embedder(partial_eq)
        seq_emb = self.pe(seq_emb)
        if timer is not None:
            torch.cuda.synchronize()
            timer.add('Token-Embedding')

        # 3. 生成因果掩码 (Causal Mask)，防止模型“偷看”未来的 Token
        # causal_mask: (seq_len, seq_len)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        # 4. Transformer Decoder
        dec_out = self.transformer_decoder(
            tgt=seq_emb,
            memory=memory_expanded,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=eq_pad_mask,
            memory_key_padding_mask=memory_mask_expanded
        )
        if timer is not None:
            torch.cuda.synchronize()
            timer.add('Transformer-Decoder')

        # 5. 输出预测
        # logits: (B_seq, seq_len, n_words)
        logits = self.fc_head(dec_out)
        if timer is not None:
            torch.cuda.synchronize()
            timer.add('FC-Head')

        return logits

    def forward(self, batch_dict, timer=None):
        """
        训练时的入口函数：无缝衔接 Dataset 的 collate_fn。
        """
        # Grpah Encoding
        memory, memory_mask = self.encode_graph(
            data_node=batch_dict["data_node"],
            data_edge=batch_dict["data_edge"],
            edge_list=batch_dict["edge_list"],
            num_nodes=batch_dict["num_nodes"],
            node_batch_idx=batch_dict["node_batch_idx"],
            timer=timer,
        )

        # --- 2. 序列解码 ---
        logits = self.decode_sequence(
            partial_eq=batch_dict["partial_eqs"],
            memory=memory,
            memory_key_padding_mask=memory_mask,
            seq2graph_idx=batch_dict["seq2graph_idx"],
            eq_pad_mask=batch_dict.get("eq_pad_mask", None),
            timer=timer,
        )

        # 在训练中，如果 partial_eq 是 [A, B, C]，我们预测的是每个位置的下一个词
        # 你可以根据训练策略取最后一个时间步的预测，或者计算整条序列的 Loss
        # 如果获取最后一步: return logits[:, -1, :] 
        return logits