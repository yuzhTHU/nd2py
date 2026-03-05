# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import warnings
import numpy as np
import networkx as nx
from typing import List, Tuple, Literal
from ... import core as nd
from .ndformer_config import NDformerConfig

class NDformerGraphGenerator:
    def __init__(self, config: NDformerConfig):
        self.min_node_num = config.min_node_num
        self.max_node_num = config.max_node_num
        self.min_edge_num = config.min_edge_num
        self.max_edge_num = config.max_edge_num

    def sample(self, topology:Literal['ER', 'BA', 'WS', 'Complete'] = None, **kwargs):
        """
        Arguments:
        - V: node num
        - topology: 'ER', 'BA', 'WS', 'Complete'
        - kwargs:
            (When topology is 'ER')
            - p: edge probability
            - directed: directed or not
            (When topology is 'BA')
            - m: number of edges to attach from a new node to existing nodes
            (When topology is 'WS')
            - k: each node is connected to k nearest neighbors in ring topology
            - p: probability of rewiring each edge
            (When topology is 'Complete')
            - None
        Return:
        - edge_list: (2, E), edge list
        - num_nodes: int, node num
        """
        if topology is None:
            topology = np.random.choice(['ER', 'BA', 'WS', 'Complete'], p=[0.3, 0.3, 0.3, 0.1])
        if topology == 'ER':
            edge_list, num_nodes = self.generate_ER_graph(**kwargs)
        elif topology == 'BA':
            edge_list, num_nodes = self.generate_BA_graph(**kwargs)
        elif topology == 'WS':
            edge_list, num_nodes = self.generate_WS_graph(**kwargs)
        elif topology == 'Complete':
            edge_list, num_nodes = self.generate_complete_graph(**kwargs)
        else:
            raise ValueError(f'Unknown graph topology {topology}')
        return edge_list, num_nodes

    def generate_ER_graph(self, V=None, E=None, directed=None):
        if V is None: V = np.random.randint(self.min_node_num, self.max_node_num)
        if E is None: E = np.random.randint(self.min_edge_num, self.max_edge_num)
        if directed is None: directed = np.random.randint(0, 2)
        p = E / (V * (V-1))
        graph = nx.erdos_renyi_graph(V, p, directed=directed)
        edge_list = np.array(graph.edges()).T.tolist()
        num_nodes = graph.number_of_nodes()
        return edge_list, num_nodes

    def generate_BA_graph(self, V=None, m=None):
        if V is None: V = np.random.randint(self.min_node_num, self.max_node_num)
        if m is None: m = np.random.choice([1, 2, 3])
        graph = nx.barabasi_albert_graph(V, m)
        edge_list = np.array(graph.edges()).T.tolist()
        num_nodes = graph.number_of_nodes()
        return edge_list, num_nodes
    
    def generate_WS_graph(self, V=None, k=None, p=None):
        if V is None: V = np.random.randint(self.min_node_num, self.max_node_num)
        if k is None: k = np.random.choice([2, 4, 6])
        if p is None: p = np.random.uniform(0.1, 0.9)
        graph = nx.watts_strogatz_graph(V, k, p)
        edge_list = np.array(graph.edges()).T.tolist()
        num_nodes = graph.number_of_nodes()
        return edge_list, num_nodes

    def generate_complete_graph(self, V=None):
        if V is None: V = np.random.randint(self.min_node_num, min(self.max_node_num, int(np.sqrt(self.max_edge_num))) + 1)
        graph = nx.complete_graph(V)
        edge_list = np.array(graph.edges()).T.tolist()
        num_nodes = graph.number_of_nodes()
        return edge_list, num_nodes


class NDformerDataGenerator:
    def __init__(self, config: NDformerConfig):
        self.min_var_val = config.min_var_val
        self.max_var_val = config.max_var_val
        self.min_coeff_val = config.min_coeff_val
        self.max_coeff_val = config.max_coeff_val
        self.min_data_num = config.min_data_num
        self.max_data_num = config.max_data_num

    def sample(
        self, 
        eqtree: nd.Symbol, 
        dist_type: Literal['GMM', 'Uniform', 'Gaussian'] = 'GMM',
        edge_list: Tuple[List[int], List[int]] = None, 
        num_nodes: int = None, 
        sample_num: int = None,
        **kwargs
    ):
        """
        Arguments:
        - eqtree: a symbolic expression tree

        Returns:
            var_dict = dict(
                A: np.ndarray, (V, V)
                G: np.ndarray, (E, 2)
                out: np.ndarray, (N, V) or (N, E)
                v1/v2/v3/v4/v5: np.ndarray, (N, V)
                e1/e2/e3/e4/e5: np.ndarray, (N, E)
            )
        """
        if sample_num is None:
            sample_num = np.random.randint(self.min_data_num, self.max_data_num + 1)
        if num_nodes is None and edge_list is not None:
            num_nodes = np.reshape(edge_list, (-1,)).max() + 1
        # 采样公式中的数值常数
        for node in eqtree.iter_preorder():
            if isinstance(node, nd.Number):
                node.value = np.random.uniform(self.min_coeff_val, self.max_coeff_val)
        # 解析要采样的变量
        variables = [var for var in eqtree.iter_preorder() if isinstance(var, nd.Variable)]
        var_dims = []
        for var in variables:
            if var.nettype == 'node':
                var_dims.append(num_nodes)
            elif var.nettype == 'edge':
                var_dims.append(len(edge_list[0]))
            elif var.nettype == 'scalar':
                var_dims.append(1)
            else:
                raise ValueError(f'Unsupported nettype: {var.nettype}')
        # 采样公式中的变量取值
        collected_vars = {var.name: [] for var in variables}
        collected_targets = []
        needed_num = sample_num
        for attempt in range(10):
            # current_batch_size = max(10, 2 * needed_num)
            current_batch_size = needed_num
            if dist_type == 'GMM':
                samples = self.generate_GMM_data(size=(current_batch_size, sum(var_dims)), **kwargs)
            elif dist_type == 'Uniform':
                samples = self.generate_uniform_data(size=(current_batch_size, sum(var_dims)), **kwargs)
            elif dist_type == 'Gaussian':
                samples = self.generate_normal_data(size=(current_batch_size, sum(var_dims)), **kwargs)
            else:
                raise ValueError(f'Unknown data generation dist_type: {dist_type}')
            split_arrays = np.split(samples, np.cumsum(var_dims)[:-1], axis=1)
            batch_vars = {var.name: split_array for var, split_array in zip(variables, split_arrays)}
            batch_target = eqtree.eval(batch_vars, edge_list=edge_list, num_nodes=num_nodes)
            return batch_vars, batch_target
            valid_mask = np.isfinite(batch_target).all(axis=-1)
            collected_targets.append(batch_target[valid_mask, :])
            for var_name in batch_vars:
                collected_vars[var_name].append(batch_vars[var_name][valid_mask])
            needed_num -= valid_mask.sum()
            if needed_num <= 0: break
        else:
            warnings.warn(f"Only collected {sample_num - needed_num} valid samples out of requested {sample_num} after 10 attempts.")
        # 将收集到的样本拼接成最终的输出
        if needed_num > 0:
            final_vars = {var: array[:sample_num] for var, array in batch_vars.items()}
            final_target = batch_target[:sample_num]
        else:
            final_vars = {var: np.concatenate(arrays, axis=0)[:sample_num] for var, arrays in collected_vars.items()}
            final_target = np.concatenate(collected_targets, axis=0)[:sample_num]
        return final_vars, final_target

    def generate_normal_data(self, size, mean=None, std=None):
        if mean is None: mean = (self.min_var_val + self.max_var_val) / 2
        if std is None: std = (self.max_var_val - self.min_var_val) / 6  # 99.7% data within [min_var_val, max_var_val]
        return np.random.normal(loc=mean, scale=std, size=size)
    
    def generate_uniform_data(self, size, low=None, high=None):
        if low is None: low = self.min_var_val
        if high is None: high = self.max_var_val
        return np.random.uniform(low=low, high=high, size=size)
    
    def generate_GMM_data(self, size, L=1):
        if not isinstance(size, tuple) or len(size) != 2:
            raise ValueError("Size must be a tuple of (N, D)")
        N, D = size
        K = np.random.randint(1, 11)
        pi = np.random.rand(K)
        sigma_Z = np.random.uniform(0.0, 10.0, (K,))
        sigma_X = np.random.uniform(0.0, 3.0, (K,))
        A = np.random.uniform(-1.0, 1.0, (D, L, K))
        b = np.random.uniform(-10.0, 10.0, (D, K))
        C = np.random.choice(K, (N,), p=pi / pi.sum())
        Z = np.random.normal(0, sigma_Z, (N, L, K))
        n = np.random.normal(0, sigma_X, (N, D, K))
        X = np.einsum('DlK,NlK->NDK', A, Z) + b + n
        return np.choose(C, X.transpose(2, 1, 0)).transpose(1, 0)
