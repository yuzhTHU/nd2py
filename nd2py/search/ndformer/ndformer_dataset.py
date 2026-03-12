# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import torch
import logging
import itertools
import copy
import warnings
import numpy as np
import torch.utils.data as D
from typing import Optional
from torch_geometric.data import Batch
from ... import core as nd
from .ndformer_config import NDFormerConfig
from .ndformer_generator import NDFormerEqtreeGenerator, NDFormerGraphGenerator, NDFormerDataGenerator
from .ndformer_tokenizer import NDFormerTokenizer

# 屏蔽 eval 结果赋值时的 overflow 警告
warnings.filterwarnings("ignore", message="overflow encountered in cast")
warnings.filterwarnings("ignore", message="invalid value encountered in cast")

_logger = logging.getLogger(f'nd2py.{__name__}')


class InfiniteSampler(D.Sampler):
    # 无限生成索引，用于 DataLoader(sampler=InfiniteSampler())
    def __iter__(self):
        return itertools.count()


class NDFormerDataset(D.Dataset):
    def __init__(
        self,
        config: NDFormerConfig,
        eqtree_generator: NDFormerEqtreeGenerator,
        topo_generator: NDFormerGraphGenerator,
        data_generator: NDFormerDataGenerator,
        tokenizer: NDFormerTokenizer,
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
        data_edge = np.zeros((sample_num, num_edges, self.config.max_var_num + 1), dtype=np.float32)
        data_scalar = np.zeros((sample_num, 1, self.config.max_var_num + 1), dtype=np.float32)
        for var in eqtree.iter_preorder():
            if isinstance(var, nd.Variable):
                var_token = self.tokenizer.variable_mapping[var.name]
                if var_token in self.tokenizer.node_var_tokens:
                    i = self.tokenizer.node_var_tokens.index(var_token)
                    data_node[:, :, i] = data_dict[var.name]
                elif var_token in self.tokenizer.edge_var_tokens:
                    i = self.tokenizer.edge_var_tokens.index(var_token)
                    data_edge[:, :, i] = data_dict[var.name]
                elif var_token in self.tokenizer.scalar_var_tokens:
                    i = self.tokenizer.scalar_var_tokens.index(var_token)
                    data_scalar[:, :, i] = data_dict[var.name]
                else:
                    raise ValueError(f'Unknown variable: {var.name}')

        # 将公式结果写入最后一个特征通道
        if eqtree.nettype == 'node':
            data_node[:, :, -1] = eqtree.eval(data_dict, edge_list=edge_list, num_nodes=num_nodes)
        elif eqtree.nettype == 'edge':
            data_edge[:, :, -1] = eqtree.eval(data_dict, edge_list=edge_list, num_nodes=num_nodes)
        elif eqtree.nettype == 'scalar':
            data_scalar[:, :, -1] = eqtree.eval(data_dict, edge_list=edge_list, num_nodes=num_nodes)
        else:
            raise ValueError(f'Unsupported nettype: {eqtree.nettype}')

        data_node = self.tokenizer.encode_array(data_node, mode='token_id')
        data_edge = self.tokenizer.encode_array(data_edge, mode='token_id')
        data_scalar = self.tokenizer.encode_array(data_scalar, mode='token_id')

        # Generate training samples using progressive subtree replacement
        # Instead of prefix splitting, we progressively replace subtrees with Empty()
        # and use the symbol at the first Empty position as next_token
        partial_eqs = []
        next_tokens = []
        # === 我修改了这一部分 ===
        eqtree_ = eqtree.copy()
        while not isinstance(eqtree_, nd.Empty):
            # 可选择被替换的节点：叶子节点 或 所有子节点都已是 Empty 的节点
            def is_replaceable(node):
                if node.n_operands == 0:
                    return True  # 叶子节点
                return all(isinstance(op, nd.Empty) for op in node.operands)  # 所有子节点都是 Empty

            candidates = [i for i in eqtree_.iter_preorder() if is_replaceable(i) and not isinstance(i, nd.Empty)]
            if not candidates:
                break
            node_to_replace = np.random.choice(candidates)
            eqtree_ = eqtree_.replace(node_to_replace, nd.Empty(), no_warn=True)
            # 找到第一个 Empty
            for sym in eqtree_.iter_preorder():
                if isinstance(sym, nd.Empty):
                    break
            # 找到 eqtree 中该 Empty 位置对应的原始符号
            empty_path = eqtree_.path_to(sym)
            original_sym = eqtree.get_path(empty_path)

            # 编码不完整的树
            tokens, parent_ids, nettype_ids = self.tokenizer.encode(eqtree_, mode='token_id')
            partial_eqs.append(torch.tensor(tokens, dtype=torch.long))

            # 获取 next_token
            if isinstance(original_sym, nd.Variable):
                next_token_str = self.tokenizer.variable_mapping[original_sym.name]
            elif isinstance(original_sym, nd.Number):
                num_tokens = self.tokenizer.num_tokenizer.encode(original_sym.value, mode='token')
                next_token_str = num_tokens[0]  # Sign token
            elif isinstance(original_sym, nd.Empty):
                next_token_str = self.tokenizer.empty_token
            else:
                next_token_str = type(original_sym).__name__

            next_token_id = self.tokenizer.token2id.get(next_token_str, self.tokenizer.unk_token_id)
            next_tokens.append(torch.tensor(next_token_id, dtype=torch.long))

        return {
            'edge_list': torch.tensor(edge_list, dtype=torch.long), # (2, EdgeNum)
            'data_node': torch.tensor(data_node, dtype=torch.long), # (SampleNum, NodeNum, max_var_num+1, 3)
            'data_edge': torch.tensor(data_edge, dtype=torch.long), # (SampleNum, EdgeNum, max_var_num+1, 3)
            'data_scalar': torch.tensor(data_scalar, dtype=torch.long), # (SampleNum, 1, max_var_num+1, 3)
            'num_nodes': num_nodes,     # int
            'partial_eqs': partial_eqs, # List of (SeqLen,)
            'next_tokens': next_tokens, # List of (1,)
        }

    def collate_fn(self, batch):
        batched_edge_list = [] # List of (2, EdgeNum_i)
        batched_data_node = [] # List of (SampleNum, NodeNum_i, max_var_num+1, 3)
        batched_data_edge = [] # List of (SampleNum, EdgeNum_i, max_var_num+1, 3)
        batched_data_scalar = [] # List of (SampleNum, 1, max_var_num+1, 3)
        batched_num_nodes = 0
        batched_partial_eqs = []
        batched_next_tokens = []
        node_batch_idx = [] # 记录每个节点属于 batch 中的哪张图，后续使用 to_dense_batch 还原时需要用到
        seq_batch_idx = [] # 记录每个 (partial_eq, next_token) 序列属于 batch 中的哪张图，后续构建 seq2graph_idx 时需要用到

        for batch_idx, data in enumerate(batch):
            batched_edge_list.append(data['edge_list'] + batched_num_nodes)
            batched_data_node.append(data['data_node'])
            batched_data_edge.append(data['data_edge'])
            batched_data_scalar.append(data['data_scalar'])
            batched_num_nodes += data['num_nodes']
            node_batch_idx.extend([batch_idx] * data['num_nodes'])
            for partial_eq, next_token in zip(data['partial_eqs'], data['next_tokens']):
                batched_partial_eqs.append(partial_eq)
                batched_next_tokens.append(next_token)
                seq_batch_idx.append(batch_idx) # 记录该序列属于当前 batch_idx

        final_edge_list = torch.cat(batched_edge_list, dim=1) # (2, TotalEdges)
        final_data_node = torch.cat(batched_data_node, dim=1) # (SampleNum, TotalNodes, max_var_num+1, 3)
        final_data_edge = torch.cat(batched_data_edge, dim=1) # (SampleNum, TotalEdges, max_var_num+1, 3)
        final_data_scalar = torch.cat(batched_data_scalar, dim=1) # (SampleNum, TotalBatchs, max_var_num+1, 3)
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
            "data_scalar": final_data_scalar, # (SampleNum, TotalBatchs, max_var_num+1, 3)
            "num_nodes": final_num_nodes, # int
            "partial_eqs": final_partial_eqs, # (TotalSeqs, MaxSeqLen)
            "next_tokens": final_next_tokens, # (TotalSeqs,)
            "node_batch_idx": final_node_batch_idx, # (TotalNodes,) data_node 中每个节点与所属图索引的映射
            "seq_batch_idx": final_seq_batch_idx    # (TotalSeqs,) partial_eqs / next_tokens 中每个序列与所属图索引的映射
        }

    def get_sampler(self):
        return InfiniteSampler() if self.n_samples is None else None
