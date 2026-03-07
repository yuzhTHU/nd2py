# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import torch
import logging
import itertools
import numpy as np
import torch.utils.data as D
from typing import Optional
from torch_geometric.data import Batch
from ... import core as nd
from .ndformer_config import NDformerConfig
from .ndformer_generator import NDformerEqtreeGenerator, NDformerGraphGenerator, NDformerDataGenerator
from .ndformer_tokenizer import NDformerTokenizer

_logger = logging.getLogger(f'nd2py.{__name__}')


class InfiniteSampler(D.Sampler):
    # 无限生成索引，用于 DataLoader(sampler=InfiniteSampler())
    def __iter__(self):
        return itertools.count()


class NDformerDataset(D.Dataset):
    def __init__(
        self, 
        config: NDformerConfig,
        eqtree_generator: NDformerEqtreeGenerator, 
        topo_generator: NDformerGraphGenerator,
        data_generator: NDformerDataGenerator, 
        tokenizer: NDformerTokenizer, 
        n_samples: Optional[int] = None, 
        random_state: Optional[int] = None
    ):
        self.config = config
        self.eqtree_generator = eqtree_generator
        self.topo_generator = topo_generator
        self.data_generator = data_generator
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.random_state = random_state

    def __len__(self):
        # 如果 n_samples 为 None, 实际的无限循环由 InfiniteSampler 接管
        return self.n_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng((self.random_state, idx)) if self.random_state is not None else None
        eqtree = self.eqtree_generator.sample(nettypes={"node", "edge", "scalar"}, _rng=rng)
        edge_list, num_nodes = self.topo_generator.sample(_rng=rng)
        data_dict, target = self.data_generator.sample(eqtree, edge_list=edge_list, num_nodes=num_nodes, sample_num=200, _rng=rng)
        num_edges = len(edge_list[0])
        sample_num = target.shape[0]

        _logger.debug(f"Sampled eqtree: {eqtree.to_str(number_format='.2f')}")
        
        vars_dict = {var.name: var for var in eqtree.iter_preorder() if isinstance(var, nd.Variable)}
        data_node = np.zeros((sample_num, num_nodes, self.config.max_var_num + 1), dtype=np.float32)
        for i, var_name in enumerate(var_name for var_name, var in vars_dict.items() if var.nettype == 'node'): # TODO!
            data_node[:, :, i] = data_dict[var_name]  # 假设 data_dict[var] 形状为 (sample_num, num_nodes)
            
        data_edge = np.zeros((sample_num, num_edges, self.config.max_var_num + 1), dtype=np.float32)
        for i, var_name in enumerate(var_name for var_name, var in vars_dict.items() if var.nettype == 'edge'):
            data_edge[:, :, i] = data_dict[var_name]  # 假设 data_dict[var] 形状为 (sample_num, num_edges)
        
        # 将公式结果写入最后一个特征通道
        if eqtree.nettype == 'node':
            data_node[:, :, -1] = eqtree.eval(data_dict, edge_list=edge_list, num_nodes=num_nodes)
        elif eqtree.nettype == 'edge':
            data_edge[:, :, -1] = eqtree.eval(data_dict, edge_list=edge_list, num_nodes=num_nodes)
        else:
            raise ValueError(f'Unsupported nettype: {eqtree.nettype}')
        
        data_node = self.tokenizer.encode_array(data_node, mode='token_id')
        data_edge = self.tokenizer.encode_array(data_edge, mode='token_id')

        tokens, _, _ = self.tokenizer.encode(eqtree, mode='token_id') 
        # 将一个完整序列 [A, B, C, D] 切分为多对 (前缀, 预测目标)
        # 例如: ([A], B), ([A, B], C), ([A, B, C], D)
        partial_eqs = []
        next_tokens = []
        for i in range(1, len(tokens)):
            partial_eqs.append(tokens[:i])
            next_tokens.append(tokens[i])

        return {
            'edge_list': torch.tensor(edge_list, dtype=torch.long), # (2, EdgeNum)
            'data_node': torch.tensor(data_node, dtype=torch.long), # (SampleNum, NodeNum, max_var_num+1, 3)
            'data_edge': torch.tensor(data_edge, dtype=torch.long), # (SampleNum, EdgeNum, max_var_num+1, 3)
            'num_nodes': num_nodes,     # int
            'partial_eqs': [torch.tensor(partial_eq, dtype=torch.long) for partial_eq in partial_eqs], # List of (SeqLen,)
            'next_tokens': [torch.tensor(next_token, dtype=torch.long) for next_token in next_tokens], # List of (1,)
        }

    def collate_fn(self, batch):
        batched_edge_list = [] # List of (2, EdgeNum_i)
        batched_data_node = [] # List of (SampleNum, NodeNum_i, max_var_num+1, 3)
        batched_data_edge = [] # List of (SampleNum, EdgeNum_i, max_var_num+1, 3)
        batched_num_nodes = 0
        batched_partial_eqs = []
        batched_next_tokens = []
        node_batch_idx = [] # 记录每个节点属于 batch 中的哪张图, 后续使用 to_dense_batch 还原时需要用到
        seq_batch_idx = [] # 记录每个 (partial_eq, next_token) 序列属于 batch 中的哪张图, 后续构建 seq2graph_idx 时需要用到

        for batch_idx, data in enumerate(batch):
            batched_edge_list.append(data['edge_list'] + batched_num_nodes)
            batched_data_node.append(data['data_node'])
            batched_data_edge.append(data['data_edge'])
            batched_num_nodes += data['num_nodes']
            node_batch_idx.extend([batch_idx] * data['num_nodes'])
            for partial_eq, next_token in zip(data['partial_eqs'], data['next_tokens']):
                batched_partial_eqs.append(partial_eq)
                batched_next_tokens.append(next_token)
                seq_batch_idx.append(batch_idx) # 记录该序列属于当前 batch_idx

        final_edge_list = torch.cat(batched_edge_list, dim=1) # (2, TotalEdges)
        final_data_node = torch.cat(batched_data_node, dim=1) # (SampleNum, TotalNodes, max_var_num+1, 3)
        final_data_edge = torch.cat(batched_data_edge, dim=1) # (SampleNum, TotalEdges, max_var_num+1, 3)
        final_num_nodes = batched_num_nodes
        final_node_batch_idx = torch.tensor(node_batch_idx, dtype=torch.long) # (TotalNodes,)
        final_partial_eqs = torch.nn.utils.rnn.pad_sequence(
            batched_partial_eqs, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        final_next_tokens = torch.tensor(batched_next_tokens, dtype=torch.long) # (TotalSeqs,)
        final_seq_batch_idx = torch.tensor(seq_batch_idx, dtype=torch.long)   # (TotalSeqs,)

        return {
            "edge_list": final_edge_list, # (2, TotalEdges)
            "data_node": final_data_node, # (SampleNum, TotalNodes, max_var_num+1, 3)
            "data_edge": final_data_edge, # (SampleNum, TotalEdges, max_var_num+1, 3)
            "num_nodes": final_num_nodes, # int
            "partial_eqs": final_partial_eqs, # (TotalSeqs, MaxSeqLen)
            "next_tokens": final_next_tokens, # (TotalSeqs,)
            "node_batch_idx": final_node_batch_idx, # (TotalNodes,) data_node 中每个节点与所属图索引的映射
            "seq_batch_idx": final_seq_batch_idx    # (TotalSeqs,) partial_eqs / next_tokens 中每个序列与所属图索引的映射
        }
    
    def get_sampler(self):
        return InfiniteSampler() if self.n_samples is None else None
