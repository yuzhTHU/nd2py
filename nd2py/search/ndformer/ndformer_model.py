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
from .ndformer_tokenizer import NDformerTokenizer
from ... import utils

# See https://github.com/pytorch/pytorch/issues/100469
warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly.")
logger = logging.getLogger('ND2.Model')


class NDformerModel(nn.Module):
    def __init__(self, config: NDformerConfig, tokenizer: NDformerTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        self.embedder = nn.Embedding(tokenizer.vocab_size, self.config.d_emb, padding_idx=tokenizer.pad_token_id)
        self.linear = nn.Linear(
            (self.config.max_var_num+1) * 3 * self.config.d_emb, 
            self.config.d_emb
        )
        
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

        self.fc_head = nn.Linear(self.config.d_emb, tokenizer.vocab_size)

    def encode_graph(self, data_node, data_edge, edge_list, num_nodes, node_batch_idx=None, timer=None):
        """
        图编码阶段：仅在图拓扑变更或初始化时调用一次。

        Args:
        - data_node: (SampleNum, NodeNum, max_var_num+1, 3)
        - data_edge: (SampleNum, EdgeNum, max_var_num+1, 3)
        - edge_list: (2, EdgeNum)
        - num_nodes: int
        - node_batch_idx: (TotalNodeNum,) 每个节点所属图的索引

        Returns:
        - memory: (BatchNum, MaxNodeNum, d_emb)
        - memory_key_padding_mask: (BatchNum, MaxNodeNum) 或 None (如果 node_batch_idx is None), 其中 True 代表需要被忽略的 Pad
        """
        # Embed
        node_emb = self.embedder(data_node).flatten(-3, -1) # (SampleNum, TotalNodeNum, (max_var_num+1)*3*d_emb)
        edge_emb = self.embedder(data_edge).flatten(-3, -1) # (SampleNum, TotalEdgeNum, (max_var_num+1)*3*d_emb)

        node_emb = self.linear(node_emb) # (SampleNum, TotalNodeNum, d_emb)
        edge_emb = self.linear(edge_emb) # (SampleNum, TotalEdgeNum, d_emb)

        if timer is not None:
            torch.cuda.synchronize()
            timer.add('Data-Embedding')

        gnn_out, _ = self.gnn_encoder(node_emb, edge_emb, edge_list, num_nodes) # (SampleNum, TotalNodeNum, d_emb)
        gnn_out = gnn_out.transpose(0, 1) # (SampleNum, TotalNodeNum, d_emb) -> (TotalNodeNum, SampleNum, d_emb)
        if timer is not None:
            torch.cuda.synchronize()
            timer.add('GNN-Encoder')

        if node_batch_idx is not None:
            (
                dense_nodes, # (BatchNum, MaxNodeNum, SampleNum, d_emb)
                valid_mask, # (BatchNum, MaxNodeNum), True 表示有效节点, False 表示 Pad
            ) = to_dense_batch(
                gnn_out, # (TotalNodeNum, SampleNum, d_emb)
                node_batch_idx, # (TotalNodeNum,)
            ) 
            batch_num, node_num, sample_num, d_emb = dense_nodes.shape
            dense_nodes = dense_nodes.flatten(1, 2) # (BatchNum, NodeNum*SampleNum, d_emb)
            valid_mask = valid_mask[..., None].expand(batch_num, node_num, sample_num).flatten(1, 2) # (BatchNum, NodeNum*SampleNum)
            src_key_padding_mask = ~valid_mask # padding_mask 中 True 代表需要被忽略的 Pad
            if timer is not None:
                torch.cuda.synchronize()
                timer.add('To-Dense-Batch')
        else:
            dense_nodes = gnn_out.unsqueeze(0).flatten(1, 2) # (1, NodeNum*SampleNum, d_emb)
            src_key_padding_mask = None

        memory = self.transformer_encoder(
            dense_nodes, # ([BatchNum,] NodeNum*SampleNum, d_emb)
            src_key_padding_mask=src_key_padding_mask, # ([BatchNum,] NodeNum*SampleNum), 其中 True 代表需要被忽略的 Pad
        ) # ([BatchNum,] NodeNum*SampleNum, d_emb)
        if timer is not None:
            torch.cuda.synchronize()
            timer.add('Transformer-Encoder')
        return memory, src_key_padding_mask

    def decode_sequence(self, memory, partial_eq, memory_key_padding_mask=None, seq_batch_idx=None, timer=None):
        """
        序列解码阶段：支持 1-to-N 广播，可高频调用。
        
        Args: 
        - memory: (BatchNum, NodeNum*SampleNum, d_emb), 来自 encode_graph 的输出
        - partial_eq: (SeqNum, MaxSeqLen)
        - memory_key_padding_mask: (BatchNum, NodeNum*SampleNum) 或 None, 其中 True 代表需要被忽略的 Pad
        - seq_batch_idx: (SeqNum,) 每个序列所属图的索引, 用于将 memory 中的节点特征正确广播到每个序列

        Returns:
        - logits: (SeqNum, vocab_size)
        """
        seq_emb = self.embedder(partial_eq) # (SeqNum, SeqLen, d_emb)
        seq_emb = self.pe(seq_emb) # (SeqNum, SeqLen, d_emb)
        tgt_key_padding_mask = (partial_eq == self.tokenizer.pad_token_id) # (SeqNum, SeqLen), True 代表需要被忽略的 Pad
        if seq_batch_idx is not None:
            memory = memory[seq_batch_idx] # (SeqNum, NodeNum*SampleNum, d_emb)
            if memory_key_padding_mask is not None:
                memory_key_padding_mask = memory_key_padding_mask[seq_batch_idx] # (SeqNum, NodeNum*SampleNum)
        else:
            memory = memory.expand(seq_emb.shape[0], -1, -1) # (SeqNum, NodeNum*SampleNum, d_emb)
            if memory_key_padding_mask is not None:
                memory_key_padding_mask = memory_key_padding_mask.expand(seq_emb.shape[0], -1) # (SeqNum, NodeNum*SampleNum)
        
        if timer is not None:
            torch.cuda.synchronize()
            timer.add('Token-Embedding')

        dec_out = self.transformer_decoder(
            tgt=seq_emb,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        if timer is not None:
            torch.cuda.synchronize()
            timer.add('Transformer-Decoder')

        logits = self.fc_head(dec_out).mean(axis=-2).log_softmax(axis=-1) # (SeqNum, SeqLen, vocab_size) -> (SeqNum, vocab_size)
        if timer is not None:
            torch.cuda.synchronize()
            timer.add('FC-Head')
        return logits

    def forward(self, batch_dict, timer=None):
        """
        训练时的入口函数：无缝衔接 Dataset 的 collate_fn。
        """
        memory, memory_key_padding_mask = self.encode_graph(
            data_node=batch_dict["data_node"], # (SampleNum, TotalNodes, max_var_num+1, 3)
            data_edge=batch_dict["data_edge"], # (SampleNum, TotalEdges, max_var_num+1, 3)
            edge_list=batch_dict["edge_list"], # (2, TotalEdges)
            num_nodes=batch_dict["num_nodes"], # int
            node_batch_idx=batch_dict.get("node_batch_idx", None), # (Batch,)
            timer=timer,
        )
        logits = self.decode_sequence( # (SeqNum, vocab_size)
            memory=memory, # (BatchNum, NodeNum*SampleNum, d_emb)
            partial_eq=batch_dict["partial_eqs"], # (SeqNum, MaxSeqLen)
            memory_key_padding_mask=memory_key_padding_mask, # (BatchNum, NodeNum*SampleNum) 或 None
            seq_batch_idx=batch_dict.get("seq_batch_idx", None), # (SeqNum,)
            timer=timer,
        )
        return logits
