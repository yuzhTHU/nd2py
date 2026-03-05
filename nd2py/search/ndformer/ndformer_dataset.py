# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import itertools
import numpy as np
import torch
import torch.utils.data as D
from typing import Optional
from ... import core as nd
from ...generator import GPLearnGenerator
from .ndformer_config import NDformerConfig
from .ndformer_generator import NDformerGraphGenerator, NDformerDataGenerator
from .ndformer_tokenizer import NDformerTokenizer

class InfiniteSampler(D.Sampler):
    # 无限生成索引，用于 DataLoader(sampler=InfiniteSampler())
    def __iter__(self):
        return itertools.count()


class NDformerDataset(D.Dataset):
    def __init__(
        self, 
        config: NDformerConfig,
        eq_generator: GPLearnGenerator, 
        topo_generator: NDformerGraphGenerator,
        data_generator: NDformerDataGenerator, 
        tokenizer: NDformerTokenizer, 
        n_samples: Optional[int] = None, 
    ):
        self.config = config
        self.eq_generator = eq_generator
        self.topo_generator = topo_generator
        self.data_generator = data_generator
        self.tokenizer = tokenizer
        self.n_samples = n_samples

    def __len__(self):
        # 如果 n_samples 为 None, 实际的无限循环由 InfiniteSampler 接管
        return self.n_samples

    def __getitem__(self, idx):
        eqtree = self.eq_generator.sample(nettypes={"node", "edge", "scalar"})
        edge_list, num_nodes = self.topo_generator.sample()
        data_dict, target = self.data_generator.sample(eqtree, edge_list=edge_list, num_nodes=num_nodes, sample_num=200)
        num_edges = len(edge_list[0])
        sample_num = target.shape[0]
        
        vars_dict = {var.name: var for var in eqtree.iter_preorder() if isinstance(var, nd.Variable)}
        data_node = np.zeros((sample_num, num_nodes, self.config.max_var_num + 1), dtype=np.float32)
        for i, var_name in enumerate(var_name for var_name, var in vars_dict.items() if var.nettype == 'node'):
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

        # ==========================================
        # 核心新增：利用 Tokenizer 切分 partial_eq
        # ==========================================
        # 假设 encode 返回的是 List[int]
        tokens, _, _ = self.tokenizer.encode(eqtree, mode='token_id') 
        # 将一个完整序列 [A, B, C, D] 切分为多对 (前缀, 预测目标)
        # 例如: ([A], B), ([A, B], C), ([A, B, C], D)
        partial_eqs = []
        next_tokens = []
        for i in range(1, len(tokens)):
            partial_eqs.append(tokens[:i])
            next_tokens.append(tokens[i])

        return {
            'edge_list': edge_list,    # (2, NodeNum)
            'data_node': data_node,    # (SampleNum, NodeNum, max_var_num)
            'data_edge': data_edge,    # (SampleNum, EdgeNum, max_var_num)
            'num_nodes': num_nodes,    # int
            'partial_eqs': partial_eqs,   # List[List[int]]
            'next_tokens': next_tokens,   # List[int]
        }

    def collate_fn(self, batch):
        batched_edge_list = []
        batched_data_node = []
        batched_data_edge = []
        
        # 记录每个节点属于 batch 中的哪张图，后续使用 to_dense_batch 还原时需要用到
        node_batch_idx = [] 
        
        batched_partial_eq = []
        batched_next_token = []
        seq2graph_idx = []

        node_offset = 0

        for graph_idx, item in enumerate(batch):
            # 1. 组装大图 (Disjoint Union)
            # 边列表的索引需要加上之前所有图的节点总数偏移量
            el = torch.tensor(item['edge_list'], dtype=torch.long) + node_offset
            batched_edge_list.append(el)

            batched_data_node.append(torch.tensor(item['data_node']))
            batched_data_edge.append(torch.tensor(item['data_edge']))

            num_nodes = item['num_nodes']
            node_batch_idx.extend([graph_idx] * num_nodes)
            node_offset += num_nodes

            # 2. 收集公式序列并构建映射桥梁
            for partial_eq in item['partial_eqs']:
                batched_partial_eq.append(torch.tensor(partial_eq, dtype=torch.long))
                seq2graph_idx.append(graph_idx) # 记录该序列属于当前 graph_idx

            for next_token in item['next_tokens']:
                batched_next_token.append(next_token)

        # ==========================================
        # 合并图数据 (纯稀疏形态，无需 Pad)
        # ==========================================
        if len(batched_edge_list) > 0 and batched_edge_list[0].numel() > 0:
            final_edge_list = torch.cat(batched_edge_list, dim=1) # (2, TotalEdges)
        else:
            final_edge_list = torch.empty((2, 0), dtype=torch.long)
            
        # 沿着 Node/Edge 的维度拼接
        final_data_node = torch.cat(batched_data_node, dim=1) # (SampleNum, TotalNodes, Dim_node)
        final_data_edge = torch.cat(batched_data_edge, dim=1) # (SampleNum, TotalEdges, Dim_edge)
        final_node_batch_idx = torch.tensor(node_batch_idx, dtype=torch.long) # (TotalNodes,)

        # ==========================================
        # 合并序列数据 (变长，需要 Pad)
        # ==========================================
        # padded_partial_eq: (B_seq, MaxLength)
        padded_partial_eq = torch.nn.utils.rnn.pad_sequence(
            batched_partial_eq, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        final_next_token = torch.tensor(batched_next_token, dtype=torch.long) # (B_seq,)
        final_seq2graph_idx = torch.tensor(seq2graph_idx, dtype=torch.long)   # (B_seq,)
        
        # 生成 Decoder 所需的 padding mask (True 表示是 padding 部分，应当被忽略)
        eq_pad_mask = (padded_partial_eq == self.tokenizer.pad_token_id)

        return {
            "edge_list": final_edge_list,
            "data_node": final_data_node,
            "data_edge": final_data_edge,
            "num_nodes": sum([item['num_nodes'] for item in batch]),  # batch 中的图数量
            "partial_eqs": padded_partial_eq,
            "next_tokens": final_next_token,
            "eq_pad_mask": eq_pad_mask,
            "node_batch_idx": final_node_batch_idx, # <--- 极度重要，GNN 跑完后用它转 Dense
            "seq2graph_idx": final_seq2graph_idx    # <--- 一对多映射的桥梁
        }
    
    def get_sampler(self):
        return InfiniteSampler() if self.n_samples is None else None
