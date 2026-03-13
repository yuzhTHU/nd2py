# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""
NDFormer-guided MCTS for Symbolic Regression

Uses a pre-trained NDFormer model to guide MCTS search via PUCK
"""
import json
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Literal
from ..mcts import MCTS, Node
from ... import core as nd
from .ndformer_config import NDFormerConfig
from .ndformer_tokenizer import NDFormerTokenizer
from .ndformer_model import NDFormerModel


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
        puct_c_puct: float = 1.0,
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
        self.puct_c_puct = puct_c_puct
        self.ndformer_topk = ndformer_topk
        self.ndformer_temperature = ndformer_temperature
        self.beam_width = beam_width
        self.device = None

        # Cached memory from encoder (per graph topology)
        self._memory = None
        self._memory_mask = None

        if self.ndformer_model is not None:
            self.device = next(self.ndformer_model.parameters()).device

    def load_ndformer(
        self,
        checkpoint: str,
        config: Optional[NDFormerConfig] = None,
        variables: Optional[List[nd.Variable]] = None,
        device: Optional[str] = None,
    ):
        """
        Load pre-trained NDFormer model and tokenizer

        Args:
            checkpoint: Path to model checkpoint
            config: NDFormerConfig, if None will use default config
            variables: List of variables for tokenizer, if None will use search variables
            device: Device to load model on, if None will auto-detect
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Load config
        if config is None:
            config = NDFormerConfig()

        # Load tokenizer
        self.ndformer_tokenizer = NDFormerTokenizer(config, variables or self.variables)

        # Load model
        self.ndformer_model = NDFormerModel(config)
        checkpoint_data = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.ndformer_model.load_state_dict(checkpoint_data['model_state_dict'])
        self.ndformer_model.to(self.device)
        self.ndformer_model.eval()

        self.logger.info(f"Loaded NDFormer from {checkpoint} on {self.device}")

    def _check_ndformer_loaded(self):
        """Check if NDFormer is loaded, raise error if not"""
        if self.ndformer_model is None or self.ndformer_tokenizer is None:
            raise ValueError(
                "NDFormer not loaded. Call __init__ with ndformer parameter "
                "or call load_ndformer(checkpoint) before fit()."
            )

    def _prepare_data(self, X: Dict[str, np.ndarray], y: np.ndarray):
        """
        Prepare data for NDFormer encoder and cache the memory

        Returns:
            data_node: (sample_num, num_nodes, max_var_num+1, 3)
            data_edge: (sample_num, num_edges, max_var_num+1, 3)
            data_scalar: (sample_num, 1, max_var_num+1, 3)
        """
        # Build variable dict with target
        vars_dict = {}
        node_vars = []
        edge_vars = []
        scalar_vars = []

        for var in self.variables:
            if var.nettype == 'node':
                node_vars.append(var)
                vars_dict[var.name] = X[var.name]
            elif var.nettype == 'edge':
                edge_vars.append(var)
                vars_dict[var.name] = X[var.name]
            elif var.nettype == 'scalar':
                scalar_vars.append(var)
                vars_dict[var.name] = X[var.name]

        sample_num = y.shape[0]
        max_var_num = self.ndformer_tokenizer.config.max_var_num

        # For scalar nettype, treat scalar variables as node variables
        if self.nettype == 'scalar':
            # For scalar problems, we have one "node" per sample with all variables concatenated
            num_nodes = len(scalar_vars) if scalar_vars else 1
            num_edges = 0
        else:
            num_nodes = self.num_nodes
            num_edges = len(self.edge_list[0]) if self.edge_list else 0

        # Prepare data_node: (sample_num, num_nodes, max_var_num+1)
        data_node = np.zeros((sample_num, num_nodes, max_var_num + 1), dtype=np.float32)
        # Prepare data_edge: (sample_num, num_edges, max_var_num+1)
        data_edge = np.zeros((sample_num, num_edges, max_var_num + 1), dtype=np.float32)
        # Prepare data_scalar: (sample_num, 1, max_var_num+1)
        data_scalar = np.zeros((sample_num, 1, max_var_num + 1), dtype=np.float32)

        if self.nettype == 'scalar':
            # For scalar problems, put scalar variables in data_scalar
            scalar_idx = 0
            for i, var in enumerate(scalar_vars):
                if i < max_var_num:
                    data_scalar[:, :, scalar_idx] = X[var.name]
                    scalar_idx += 1
            # Write target to last channel
            data_scalar[:, :, -1] = y
        else:
            # For node/edge problems, fill variable values
            for i, var in enumerate(node_vars):
                if i < max_var_num:
                    data_node[:, :, i] = X[var.name]
            for i, var in enumerate(edge_vars):
                if i < max_var_num:
                    data_edge[:, :, i] = X[var.name]

            # Write target to last channel
            if self.nettype == 'node':
                data_node[:, :, -1] = y
            elif self.nettype == 'edge':
                data_edge[:, :, -1] = y

        # Encode to token IDs
        data_node = self.ndformer_tokenizer.encode_array(data_node, mode='token_id')
        data_edge = self.ndformer_tokenizer.encode_array(data_edge, mode='token_id')
        data_scalar = self.ndformer_tokenizer.encode_array(data_scalar, mode='token_id')

        return data_node, data_edge, data_scalar

    def _encode_and_cache_memory(self, X: Dict[str, np.ndarray], y: np.ndarray):
        """
        Encode graph data using NDFormer encoder and cache memory for reuse
        """
        self._check_ndformer_loaded()

        data_node, data_edge, data_scalar = self._prepare_data(X, y)

        # Convert to torch tensors
        # data_node: (sample_num, num_nodes, max_var_num+1, 3)
        # data_edge: (sample_num, num_edges, max_var_num+1, 3)
        # data_scalar: (sample_num, 1, max_var_num+1, 3)
        data_node = torch.from_numpy(data_node).to(self.device)
        data_edge = torch.from_numpy(data_edge).to(self.device)
        data_scalar = torch.from_numpy(data_scalar).to(self.device)

        # Prepare node_batch_idx for to_dense_batch
        # Since we have a single graph, all nodes belong to graph 0
        node_batch_idx = torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)

        # Call encoder
        with torch.no_grad():
            self._memory, self._memory_mask = self.ndformer_model.encode_graph(
                data_node=data_node,
                data_edge=data_edge,
                data_scalar=data_scalar,
                edge_list=torch.tensor(self.edge_list, dtype=torch.long, device=self.device),
                num_nodes=self.num_nodes,
                node_batch_idx=node_batch_idx,
            )

        self.logger.debug(f"Cached memory shape: {self._memory.shape}")

    def _get_policy_prior(
        self,
        states: List[Node],
        valid_actions_list: List[List[Tuple[nd.Symbol, nd.Symbol]]]
    ) -> List[Dict[Tuple[nd.Symbol, nd.Symbol], float]]:
        """
        Get prior probabilities from NDFormer for valid actions
        by decoding the current partial sequences in batch

        Args:
            states: List of MCTS nodes
            valid_actions_list: List of valid (empty, operator) tuples for each node

        Returns:
            List of dictionaries mapping actions to prior probabilities (one per node)
        """
        self._check_ndformer_loaded()

        # Encode current equation trees to tokens
        # Strip Identity wrapper for encoding (Identity is just a sentinel for MCTS root)
        partial_eqs = []
        for state in states:
            eqtree = state.eqtree
            if isinstance(eqtree, nd.Identity):
                # Strip Identity wrapper: copy Identity's operand to create a standalone partial tree
                eqtree = eqtree.operands[0].copy()
            tokens, _, _ = self.ndformer_tokenizer.encode(eqtree, mode='token_id')
            partial_eqs.append(torch.tensor(tokens, dtype=torch.long, device=self.device))

        # Pad sequences for batch processing
        max_len = max(len(eq) for eq in partial_eqs)
        padded_eqs = torch.stack([
            torch.cat([eq, torch.full((max_len - len(eq),), self.ndformer_tokenizer.pad_token_id, dtype=torch.long, device=self.device)])
            for eq in partial_eqs
        ])  # (batch_size, max_seq_len)

        # Get policy from decoder (predict next token)
        with torch.no_grad():
            # decode_sequence returns logits for each position
            # We only need the last position's logits to predict next token
            logits = self.ndformer_model.decode_sequence(
                partial_eq=padded_eqs,
                memory=self._memory,
                memory_key_padding_mask=self._memory_mask,
            )
            # logits shape: (batch_size, vocab_size) after mean pooling in decode_sequence
            # Apply temperature scaling
            logits = logits / self.ndformer_temperature
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)

        # Map token probabilities to action probabilities for each node
        results = []
        for i, (state, valid_actions) in enumerate(zip(states, valid_actions_list)):
            action_probs = {}
            for action in valid_actions:
                _, op = action
                op_token = type(op).__name__
                if op_token in self.ndformer_tokenizer.token2id:
                    token_id = self.ndformer_tokenizer.token2id[op_token]
                    action_probs[action] = probs[i, token_id].item()
                else:
                    action_probs[action] = 1e-6

            # Normalize
            total = sum(action_probs.values())
            if total > 0:
                action_probs = {k: v / total for k, v in action_probs.items()}
            else:
                action_probs = {a: 1.0 / len(valid_actions) for a in valid_actions}

            results.append(action_probs)

        return results

    def PUCT(self, node: Node) -> float:
        """
        PUCT score for a node

        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(sum(N) / (1 + N(s, a)))
        """
        if node.parent is None:
            return float("inf")

        prior = getattr(node, 'policy_prior', 1.0)
        exploration = self.puct_c_puct * prior * np.sqrt(
            node.parent.N / (1 + node.N)
        )

        return node.Q / (node.N + 1e-6) + exploration

    def select(self, root: Node) -> List[Node]:
        """
        Select leaf nodes using Beam Search with PUCT

        Returns a list of leaf nodes to expand in batch
        """
        # Use beam search to find top-k leaf nodes
        # Start from root and expand beam width nodes at each level
        beam = [(root, [])]  # (node, path_from_root)

        for _ in range(self.max_len):  # Max depth limit
            next_beam = []
            for node, path in beam:
                if node.children:
                    # Add all children to candidate pool
                    for i, child in enumerate(node.children):
                        next_beam.append((child, path + [i]))
                else:
                    # This is a leaf node
                    next_beam.append((node, path))

            if not next_beam:
                break

            # Score all candidates using PUCT
            scored = []
            for node, path in next_beam:
                if node.children:
                    # Internal node: use max child PUCT as proxy
                    score = max(self.PUCT(child) for child in node.children)
                else:
                    # Leaf node: use parent's PUCT toward this node
                    score = self.PUCT(node) if node.parent else 0
                scored.append((score, node, path))

            # Keep top-k nodes
            scored.sort(key=lambda x: x[0], reverse=True)
            beam = [(node, path) for _, node, path in scored[:self.beam_width]]

            # If all nodes in beam are leaves, we're done
            if all(not node.children for node, _ in beam):
                break

        # Return only leaf nodes
        return [node for node, _ in beam if not node.children]

    def expand(self, nodes: List[Node], X: Dict[str, np.ndarray], y: np.ndarray) -> List[Node]:
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
        valid_actions_dict = {}
        nodes_with_actions = []
        for node in nodes:
            valid_actions = list(self.iter_valid_action(node, shuffle=False))
            if valid_actions:
                valid_actions_dict[node] = valid_actions
                nodes_with_actions.append(node)

        if not nodes_with_actions:
            return nodes

        # Get policy priors in batch
        actions_list = [valid_actions_dict[node] for node in nodes_with_actions]
        policy_priors_list = self._get_policy_prior(nodes_with_actions, actions_list)

        # Create child nodes with priors
        selected_children = []
        for node, valid_actions, policy_priors in zip(nodes_with_actions, actions_list, policy_priors_list):
            # Sort actions by prior and select top-k
            sorted_actions = sorted(valid_actions, key=lambda a: policy_priors[a], reverse=True)
            top_actions = sorted_actions[:self.ndformer_topk]

            # Create child nodes
            for idx, action in enumerate(top_actions):
                child = self.action(node, action)
                child.parent = node
                child.xchild = len(node.children)
                child.policy_prior = policy_priors[action]
                node.children.append(child)
                if self.child_num and idx + 1 >= self.child_num:
                    break

            # Select best child for simulation
            if node.children:
                selected_children.append(max(node.children, key=self.PUCT))

        return selected_children

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
        self.MC_tree = Node(nd.Identity(nd.Empty(), nettype=self.nettype))

        # Encode graph and cache memory before MCTS search
        self._encode_and_cache_memory(X, y)

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
                (self.logger.note if _update_best else self.logger.info)(msg)

            self.records.append(record)
            if self.save_path:
                with open(self.save_path / 'records.jsonl', "a") as f:
                    f.write(json.dumps(record) + "\n")

            if _early_stop:
                self.logger.note(f"Early stop at iter {iter}")
                break
