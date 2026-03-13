# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""
NDFormer-guided MCTS for Symbolic Regression

Uses a pre-trained NDFormer model to guide MCTS search via PUCK
"""
import re
import json
import time
import torch
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Literal
from ..mcts import MCTS, Node
from ... import core as nd
from .ndformer_config import NDFormerConfig
from .ndformer_tokenizer import NDFormerTokenizer
from .ndformer_model import NDFormerModel

_logger = logging.getLogger(f'nd2py.{__name__}')


class NDFormerNode(Node):
    def __init__(self, eqtree: nd.Symbol):
        super().__init__(eqtree)
        self.policy_prior = 1.0

    def UCT(self, c) -> float:
        """
        PUCT score for a node

        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(sum(N) / (1 + N(s, a)))
        """
        if self.parent is None:
            return float("inf")
        exploration = self.policy_prior * np.sqrt(self.parent.N / (1 + self.N))
        return self.Q / (self.N + 1e-6) + c * exploration


class NDFormerMCTS(MCTS):
    """
    NDFormer-guided MCTS using PUCT for action selection

    The PUCT formula:
        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(sum(N(s, b)) / (1 + N(s, a)))

    where P(s, a) is the prior probability from NDFormer
    """

    def __init__(
        self,
        variables: List[nd.Variable],
        binary: List[nd.Symbol] = [nd.Add, nd.Sub, nd.Mul, nd.Div, nd.Max, nd.Min],
        unary: List[nd.Symbol] = [nd.Sqrt, nd.Log, nd.Abs, nd.Neg, nd.Inv, nd.Sin, nd.Cos, nd.Tan],
        max_params: int = 2,
        const_range: Tuple[float, float] = (-1.0, 1.0),
        depth_range: Tuple[int, int] = (2, 6),
        nettype: Optional[Literal["node", "edge", "scalar"]] = "scalar",
        log_per_iter: int = float("inf"),
        log_per_sec: float = float("inf"),
        log_detailed_speed: bool = False,
        save_path: str = None,
        random_state: Optional[int] = None,
        n_iter: int = 100,
        use_tqdm: bool = False,
        edge_list: Tuple[List[int], List[int]] = None,
        num_nodes: int = None,
        time_limit: float = None,
        sample_num: int = 300,
        keep_vars: bool = False,
        normalize_y: bool = False,
        normalize_X: bool = False,
        remove_abnormal: bool = False,
        train_eval_split: float = 1.0,

        child_num: int = 50,
        n_playout: int = 100,
        d_playout: int = 10,
        max_len: int = 30,
        c: float = 1.41,
        eta: float = 0.999,

        # NDFormer parameters
        ndformer: Optional[NDFormerModel] = None,
        ndformer_tokenizer: Optional[NDFormerTokenizer] = None,
        ndformer_topk: int = 10,
        ndformer_temperature: float = 1.0,
        beam_width: int = 10,  # Number of leaf nodes to expand in batch
        **kwargs,
    ):
        super().__init__(
            variables=variables,
            binary=binary,
            unary=unary,
            max_params=max_params,
            const_range=const_range,
            depth_range=depth_range,
            nettype=nettype,
            log_per_iter=log_per_iter,
            log_per_sec=log_per_sec,
            log_detailed_speed=log_detailed_speed,
            save_path=save_path,
            random_state=random_state,
            n_iter=n_iter,
            use_tqdm=use_tqdm,
            edge_list=edge_list,
            num_nodes=num_nodes,
            time_limit=time_limit,
            sample_num=sample_num,
            keep_vars=keep_vars,
            normalize_y=normalize_y,
            normalize_X=normalize_X,
            remove_abnormal=remove_abnormal,
            train_eval_split=train_eval_split,
            child_num=child_num,
            n_playout=n_playout,
            d_playout=d_playout,
            max_len=max_len,
            c=c,
            eta=eta,
            **kwargs,
        )

        # NDFormer parameters
        self.ndformer_model = ndformer
        self.ndformer_tokenizer = ndformer_tokenizer
        self.ndformer_topk = ndformer_topk
        self.ndformer_temperature = ndformer_temperature
        self.beam_width = beam_width
        self.device = None

        # Cached memory from encoder (per graph topology)
        self._memory = None
        self._memory_mask = None

        if self.ndformer_model is not None:
            self.device = next(self.ndformer_model.parameters()).device

    def fit(
        self,
        X: np.ndarray | pd.DataFrame | Dict[str, np.ndarray],
        y: np.ndarray | pd.Series,
    ):
        """
        Fit the model using NDFormer-guided MCTS with batch expansion

        First encodes the graph data and caches memory, then runs MCTS search
        with beam search based select and batch expand for efficiency
        """
        if self.ndformer_model is None or self.ndformer_tokenizer is None:
            raise ValueError(
                "NDFormer not loaded. Call __init__ with ndformer parameter "
                "or call load_ndformer(checkpoint) before fit()."
            )

        # Convert X to dict format
        if isinstance(X, np.ndarray):
            X = {var.name: x for var, x in zip(self.variables, X[..., :])}
        elif isinstance(X, pd.DataFrame):
            X = {col: X[col].values for i, col in enumerate(X.columns)}
        elif isinstance(X, dict):
            X = {k: np.asarray(v) for k, v in X.items()}
        else:
            raise ValueError(f"Unknown type: {type(X)}")

        # Root Node
        self.MC_tree = NDFormerNode(nd.Identity(nd.Empty(), nettype=self.nettype))

        # Encode graph and cache memory before MCTS search
        self.encode_data(X, y)

        # Search with batch expansion
        stop = False
        best_node = None
        self.eqtree = None
        self.named_timer.clear(reset_last_add_time=True)
        self.start_time = time.time()

        for iter in tqdm(range(1, self.n_iter + 1), disable=not self.use_tqdm):
            # Select multiple leaf nodes using beam search
            leaf_nodes = self.select(self.MC_tree)

            # Expand all selected nodes in batch
            expanded_nodes = self.expand(leaf_nodes, X, y)

            # Simulate and backpropagate for each expanded node
            iter_best = None
            for expanded in expanded_nodes:
                reward, best_simulated = self.simulate(expanded, X, y)
                self.backpropagate(expanded, reward)
                if iter_best is None or best_simulated.reward > iter_best.reward:
                    iter_best = best_simulated

            self.para_timer.add('iteration')

            ## Prepare log & record
            record = dict(
                iter=iter,
                time=self.para_timer.time,
                speed=self.para_timer.named_speed,
                time_usage=self.para_timer.named_time,
                call_count=self.para_timer.named_count,
            )

            log = {"Iter": iter}

            # Use the best simulated node from this iteration
            if iter_best is not None:
                if _update_best := (best_node is None or iter_best.reward > best_node.reward):
                    best_node = iter_best
                    self.eqtree = best_node.fitted_eqtree

                    record["eqtree"] = str(best_node.eqtree)
                    record["fitted_eqtree"] = str(best_node.fitted_eqtree)
                    record["complexity"] = best_node.complexity
                    record["reward"] = best_node.reward
                    record["r2"] = best_node.r2

                    log['Eqtree'] = str(best_node.eqtree)
                    log['Fitted eqtree'] = str(best_node.fitted_eqtree)
                    log['Complexity'] = best_node.complexity
                    log['Reward'] = f"{best_node.reward:.5f}"
                    log['R2'] = f"{best_node.r2:.5f}"

            _early_stop = (
                (iter == self.n_iter)
                or self.time_limit and (self.para_timer.time > self.time_limit)
                or best_node and best_node.r2 >= 0.99999
            )

            if (
                _update_best or _early_stop
                or self.log_per_iter and (not iter % self.log_per_iter)
                or self.log_per_sec and ('_last_log_time' not in locals() or time.time() - _last_log_time > self.log_per_sec)
            ):
                log["Speed"] = self.para_timer.to_str('time', 'speed', None)
                log['Time Usage'] = self.named_timer.to_str('pace', 'time', 'by_time')
                msg = " | ".join(f"\033[4m{k}\033[0m={v}" for k, v in log.items())
                (_logger.note if _update_best else _logger.info)(msg)

            self.records.append(record)
            if self.save_path:
                with open(self.save_path / 'records.jsonl', "a") as f:
                    f.write(json.dumps(record) + "\n")

            if _early_stop:
                _logger.note(f"Early stop at iter {iter}")
                break

    def load_ndformer(
        self,
        checkpoint_path: str = 'hf://YuMeow/ndformer:best.pth',
        config: Optional[NDFormerConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Load pre-trained NDFormer model and tokenizer

        Args:
            checkpoint_path: Path to model checkpoint. Can be:
                - Local file path: "/path/to/checkpoint.pth"
                - HF shorthand: "YuMeow/ndformer:best.pth"
                - HF full syntax: "hf://YuMeow/ndformer:best.pth"
            config: NDFormerConfig, if None will use default config
            device: Device to load model on, if None will auto-detect
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if config is None:
            config = NDFormerConfig()

        # Match HF syntax: "repo_id:filename" or "hf://repo_id:filename"
        if re.match(r'(hf://)?[\w\-]+/[\w\-]+:.*', checkpoint_path):
            from huggingface_hub import hf_hub_download
            hf_path = checkpoint_path.removeprefix('hf://')
            repo_id, filename = hf_path.split(":", 1)
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
            _logger.info(f"Downloaded {filename} from Hugging Face Hub {repo_id} into {checkpoint_path}")
        else:
            _logger.info(f"Loading local checkpoint: {checkpoint_path}")

        self.device = device
        self.ndformer_tokenizer = NDFormerTokenizer(config, self.variables)
        self.ndformer_model = NDFormerModel(config, self.ndformer_tokenizer)
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.ndformer_model.load_state_dict(checkpoint_data['model'])
        self.ndformer_model.to(self.device)
        self.ndformer_model.eval()
        _logger.info(
            f"Loaded NDFormer from {checkpoint_path} on {self.device}, "
            f"Model parameters: {sum(p.numel() for p in self.ndformer_model.parameters()):,}"
        )

    def _prepare_data(self, X: Dict[str, np.ndarray], y: np.ndarray):
        """
        Prepare data for NDFormer encoder and cache the memory

        Returns:
            data_node: (sample_num, num_nodes, max_var_num+1, 3)
            data_edge: (sample_num, num_edges, max_var_num+1, 3)
            data_scalar: (sample_num, 1, max_var_num+1, 3)
        """
        sample_num = y.shape[0]
        max_var_num = self.ndformer_tokenizer.config.max_var_num

        if self.num_nodes is None:
            num_nodes = 1
            num_edges = 0
        else:
            num_nodes = self.num_nodes
            num_edges = len(self.edge_list[0])

        data_node = np.zeros((sample_num, num_nodes, max_var_num + 1), dtype=np.float32)
        data_edge = np.zeros((sample_num, num_edges, max_var_num + 1), dtype=np.float32)
        data_scalar = np.zeros((sample_num, 1, max_var_num + 1), dtype=np.float32)
        for var in self.variables:
            var_token = self.ndformer_tokenizer.variable_mapping[var.name]
            if var.nettype == 'node':
                i = self.ndformer_tokenizer.node_var_tokens.index(var_token)
                data_node[:, :, i] = X[var.name]
            elif var.nettype == 'edge':
                i = self.ndformer_tokenizer.edge_var_tokens.index(var_token)
                data_edge[:, :, i] = X[var.name]
            elif var.nettype == 'scalar':
                i = self.ndformer_tokenizer.scalar_var_tokens.index(var_token)
                data_scalar[:, :, i] = X[var.name]
            else:
                raise ValueError(f"Unsupported nettype: {var.name}.nettype == {var.nettype}")

        # Write target to last channel
        if self.nettype == 'node':
            data_node[:, :, -1] = y
        elif self.nettype == 'edge':
            data_edge[:, :, -1] = y
        elif self.nettype == 'scalar':
            data_scalar[:, :, -1] = y
        else:
            raise ValueError(f'Unsupported nettype: self.nettype == {self.nettype}')

        # Encode to token IDs
        data_node = self.ndformer_tokenizer.encode_array(data_node, mode='token_id')
        data_edge = self.ndformer_tokenizer.encode_array(data_edge, mode='token_id')
        data_scalar = self.ndformer_tokenizer.encode_array(data_scalar, mode='token_id')
        return data_node, data_edge, data_scalar

    def encode_data(self, X: Dict[str, np.ndarray], y: np.ndarray):
        """
        Encode graph data using NDFormer encoder and cache memory for reuse
        """
        data_node, data_edge, data_scalar = self._prepare_data(X, y)
        with torch.no_grad():
            self._memory, self._memory_mask = self.ndformer_model.encode_graph(
                data_node=torch.from_numpy(data_node).to(self.device), # (sample_num, num_nodes, max_var_num+1, 3)
                data_edge=torch.from_numpy(data_edge).to(self.device), # (sample_num, num_edges, max_var_num+1, 3)
                data_scalar=torch.from_numpy(data_scalar).to(self.device), # (sample_num, 1, max_var_num+1, 3)
                edge_list=torch.tensor(self.edge_list, dtype=torch.long, device=self.device),
                num_nodes=self.num_nodes,
                # node_batch_idx=torch.zeros(self.num_nodes, dtype=torch.long, device=self.device),
            )
        _logger.debug(f"Cached memory shape: {self._memory.shape}")

    def set_policy_prior(
        self, actions_dict: Dict[NDFormerNode, List[Tuple[nd.Symbol, nd.Symbol]]],
    ):
        """
        Set prior probabilities from NDFormer for valid actions
        by decoding the current partial sequences in batch

        Args:
            states: List of MCTS nodes
            actions_dict: List of valid (empty, operator) tuples for each node

        Returns:
            List of dictionaries mapping actions to prior probabilities (one per node)
        """
        # Encode current equation trees to tokens
        partial_eqs = []
        for node in actions_dict.keys():
            eqtree = node.eqtree
            if isinstance(eqtree, nd.Identity):
                eqtree = eqtree.operands[0].copy() # Strip Identity wrapper
            tokens, _, _ = self.ndformer_tokenizer.encode(eqtree, mode='token_id')
            partial_eqs.append(torch.tensor(tokens, dtype=torch.long, device=self.device))
        padded_eqs = torch.nn.utils.rnn.pad_sequence(
            partial_eqs, batch_first=True, padding_value=self.ndformer_tokenizer.pad_token_id
        ) # (batch_size, max_seq_len)

        # Get policy from decoder (predict next token)
        with torch.no_grad():
            # decode_sequence returns logits for each position
            # We only need the last position's logits to predict next token
            logits = self.ndformer_model.decode_sequence(
                partial_eq=padded_eqs,
                memory=self._memory,
                memory_key_padding_mask=self._memory_mask,
            ) # (batch_size, vocab_size)
            probs = torch.softmax(logits / self.ndformer_temperature, dim=-1)  # (batch_size, vocab_size)

        # Map token probabilities to action probabilities for each node's children
        for node_idx, (node, actions) in enumerate(actions_dict.items()):
            for child_idx, (_, op) in enumerate(actions):
                if (op_token := type(op).__name__) in self.ndformer_tokenizer.token2id:
                    token_id = self.ndformer_tokenizer.token2id[op_token]
                    policy_prior = probs[node_idx, token_id].item()
                else:
                    policy_prior = 1e-6
                node.children[child_idx].policy_prior = policy_prior
            # Normalize
            if (total := sum(child.policy_prior for child in node.children)) > 0:
                for child in node.children:
                    child.policy_prior /= total
            else:
                for child in node.children:
                    child.policy_prior = 1.0 / len(node.children)

    def select(self, root: NDFormerNode) -> List[NDFormerNode]:
        """
        Select leaf nodes using Beam Search with PUCT

        Returns a list of leaf nodes to expand in batch
        """
        # Use beam search to find top-k leaf nodes
        # Start from root and expand beam width nodes at each level
        beam = [root]
        for _ in range(self.max_len):  # Max depth limit
            next_beam = []
            for node in beam:
                if node.children:
                    next_beam.extend(node.children) # Add all children to candidate pool
                else:
                    next_beam.append(node) # This is a leaf node
            next_beam.sort(key=lambda x: x.UCT(self.c), reverse=True)
            beam = next_beam[:self.beam_width]
            if all(not node.children for node in beam):
                break # If all nodes in beam are leaves, we're done
        # Return only leaf nodes
        return [node for node in beam if not node.children]

    def expand(self, nodes: List[NDFormerNode], X: Dict[str, np.ndarray], y: np.ndarray) -> List[NDFormerNode]:
        """
        Expand multiple nodes with NDFormer-guided action selection in batch

        Args:
            nodes: List of leaf nodes to expand

        Returns:
            List of selected child nodes for simulation
        """
        if not nodes:
            return []

        # Collect valid actions for each node
        actions_dict = defaultdict(list) # node -> list of actions
        for node in nodes:
            for idx, action in enumerate(self.iter_valid_action(node, shuffle=False)):
                child = self.action(node, action)
                child.parent = node
                node.children.append(child)
                actions_dict[node].append(action)
        if not actions_dict: # i.e., not any (node.children for node in nodes)
            return nodes # all nodes are leaf nodes
        self.set_policy_prior(actions_dict)
        return [random.choice(node.children) if node.children else node for node in nodes]
