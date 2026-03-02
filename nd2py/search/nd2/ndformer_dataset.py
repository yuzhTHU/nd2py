import itertools
import numpy as np
import torch
import torch.utils.data as D
from typing import Optional

class InfiniteSampler(D.Sampler):
    # 无限生成索引，用于 DataLoader(sampler=InfiniteSampler())
    def __iter__(self):
        return itertools.count()

class NDformerDataset(D.Dataset):
    def __init__(self, eq_generator, data_generator, tokenizer, max_dim_node: int = 10, max_dim_edge: int = 10, n_samples: Optional[int] = None):
        self.eq_generator = eq_generator
        self.data_generator = data_generator
        self.tokenizer = tokenizer
        self.max_dim_node = max_dim_node
        self.max_dim_edge = max_dim_edge
        self.n_samples = n_samples
    
    def __len__(self):
        # 如果 n_samples 为 None, 实际的无限循环由 InfiniteSampler 接管
        return self.n_samples

    def __getitem__(self, idx):
        sample_num = 200
        eq = self.eq_generator.sample()
        variables = eq.get_variables()
        edge_list, num_nodes, data_dict = self.data_generator.sample(eq, sample_num=sample_num)
        num_edges = edge_list.shape[1]
        
        data_node = np.zeros((sample_num, num_nodes, self.max_dim_node), dtype=np.float32)
        for i, var in enumerate(var for var in variables if var.nettype == 'node'):
            data_node[:, :, i] = data_dict[var]  # 假设 data_dict[var] 形状为 (sample_num, num_nodes)
            
        data_edge = np.zeros((sample_num, num_edges, self.max_dim_edge), dtype=np.float32)
        for i, var in enumerate(var for var in variables if var.nettype == 'edge'):
            data_edge[:, :, i] = data_dict[var]  # 假设 data_dict[var] 形状为 (sample_num, num_edges)
        
        # 将公式结果写入最后一个特征通道
        if eq.nettype == 'node':
            data_node[:, :, -1] = eq.eval(data_dict, edge_list=edge_list, num_nodes=num_nodes)
        elif eq.nettype == 'edge':
            data_edge[:, :, -1] = eq.eval(data_dict, edge_list=edge_list, num_nodes=num_nodes)
        else:
            raise ValueError(f'Unsupported nettype: {eq.nettype}')

        # ==========================================
        # 核心新增：利用 Tokenizer 切分 partial_eq
        # ==========================================
        # 假设 encode 返回的是 List[int]
        tokens = self.tokenizer.encode(eq) 
        eq_samples = []
        # 将一个完整序列 [A, B, C, D] 切分为多对 (前缀, 预测目标)
        # 例如: ([A], B), ([A, B], C), ([A, B, C], D)
        for i in range(1, len(tokens)):
            partial_eq = tokens[:i]
            next_token = tokens[i]
            eq_samples.append((partial_eq, next_token))

        return {
            'edge_list': edge_list,    # (2, NodeNum)
            'data_node': data_node,    # (SampleNum, NodeNum, max_dim_node)
            'data_edge': data_edge,    # (SampleNum, EdgeNum, max_dim_edge)
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'eq_samples': eq_samples   # List[Tuple[List[int], int]]
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
            for partial_eq, next_token in item['eq_samples']:
                batched_partial_eq.append(torch.tensor(partial_eq, dtype=torch.long))
                batched_next_token.append(next_token)
                seq2graph_idx.append(graph_idx) # 记录该序列属于当前 graph_idx

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
            "node_batch_idx": final_node_batch_idx, # <--- 极度重要，GNN 跑完后用它转 Dense
            "partial_eq": padded_partial_eq,
            "next_token": final_next_token,
            "eq_pad_mask": eq_pad_mask,
            "seq2graph_idx": final_seq2graph_idx    # <--- 一对多映射的桥梁
        }
    
    def get_sampler(self):
        return InfiniteSampler() if self.n_samples is None else None