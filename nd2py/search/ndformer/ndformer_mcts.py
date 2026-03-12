# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""
NDFormer-guided MCTS for Symbolic Regression

Uses a pre-trained NDFormer model to guide MCTS search via PUCT
"""
import torch
import numpy as np
import pandas as pd
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
            data_node: (1, total_nodes, max_var_num+1)
            data_edge: (1, total_edges, max_var_num+1)
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

        if self.nettype == 'scalar':
            # For scalar problems, put each scalar variable at a different "node" position
            for i, var in enumerate(scalar_vars):
                if i < max_var_num:
                    data_node[:, i, :] = 0  # Reset
                    data_node[:, i, i] = X[var.name]  # Put variable value in diagonal position
            # Write target to last channel of first node
            data_node[:, 0, -1] = y
        else:
            # For node/edge problems, fill variable values
            for i, var in enumerate(node_vars):
                if i < max_var_num:
                    data_node[:, :, i] = X[var.name]

        # Prepare data_edge: (sample_num, num_edges, max_var_num+1)
        data_edge = np.zeros((sample_num, num_edges, max_var_num + 1), dtype=np.float32)
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

        return data_node, data_edge

    def _encode_and_cache_memory(self, X: Dict[str, np.ndarray], y: np.ndarray):
        """
        Encode graph data using NDFormer encoder and cache memory for reuse
        """
        self._check_ndformer_loaded()

        data_node, data_edge = self._prepare_data(X, y)

        # Convert to torch tensors
        # data_node: (sample_num, num_nodes, max_var_num+1, 3)
        # data_edge: (sample_num, num_edges, max_var_num+1, 3)
        data_node = torch.from_numpy(data_node).to(self.device)
        data_edge = torch.from_numpy(data_edge).to(self.device)

        # For scalar nettype, skip graph encoding and use simpler encoding
        if self.nettype == 'scalar':
            # For scalar problems, we don't have graph structure
            # Use transformer encoder directly on node embeddings
            with torch.no_grad():
                # Embed nodes
                node_emb = self.ndformer_model.embedder(data_node).flatten(-3, -1)
                node_emb = self.ndformer_model.linear(node_emb)

                # Add positional encoding
                node_emb = self.ndformer_model.pe(node_emb)

                # Pass through transformer encoder
                memory = self.ndformer_model.transformer_encoder(node_emb)
                self._memory = memory  # (sample_num, num_nodes, d_emb)
                self._memory_mask = None
        else:
            # Prepare node_batch_idx for to_dense_batch
            # Since we have a single graph, all nodes belong to graph 0
            node_batch_idx = torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)

            # Call encoder
            with torch.no_grad():
                self._memory, self._memory_mask = self.ndformer_model.encode_graph(
                    data_node=data_node,
                    data_edge=data_edge,
                    edge_list=torch.tensor(self.edge_list, dtype=torch.long, device=self.device),
                    num_nodes=self.num_nodes,
                    node_batch_idx=node_batch_idx,
                )

        self.logger.debug(f"Cached memory shape: {self._memory.shape}")

    def _get_policy_prior(
        self,
        state: Node,
        valid_actions: List[Tuple[nd.Symbol, nd.Symbol]]
    ) -> Dict[Tuple[nd.Symbol, nd.Symbol], float]:
        """
        Get prior probabilities from NDFormer for valid actions
        by decoding the current partial sequence

        Args:
            state: Current MCTS node
            valid_actions: List of valid (empty, operator) tuples

        Returns:
            Dictionary mapping actions to prior probabilities
        """
        self._check_ndformer_loaded()

        # Encode current equation tree to tokens
        tokens, _, _ = self.ndformer_tokenizer.encode(state.eqtree, mode='token_id')

        # Add SOS token at the beginning if needed
        # Looking at tokenizer, sos_token_id is used for sequence start
        # For prediction, we need to predict next token after current sequence
        partial_eq = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Get policy from decoder (predict next token)
        with torch.no_grad():
            # decode_sequence returns logits for each position
            # We only need the last position's logits to predict next token
            logits = self.ndformer_model.decode_sequence(
                partial_eq=partial_eq,
                memory=self._memory,
                memory_key_padding_mask=self._memory_mask,
            )
            # logits shape: (1, vocab_size) after mean pooling in decode_sequence
            # Apply temperature scaling
            last_logits = logits[0] / self.ndformer_temperature
            probs = torch.softmax(last_logits, dim=-1)  # (vocab_size,)

        # Map token probabilities to action probabilities
        action_probs = {}
        for action in valid_actions:
            _, op = action
            op_token = type(op).__name__
            if op_token in self.ndformer_tokenizer.token2id:
                token_id = self.ndformer_tokenizer.token2id[op_token]
                action_probs[action] = probs[token_id].item()
            else:
                action_probs[action] = 1e-6

        # Normalize
        total = sum(action_probs.values())
        if total > 0:
            action_probs = {k: v / total for k, v in action_probs.items()}
        else:
            action_probs = {a: 1.0 / len(valid_actions) for a in valid_actions}

        return action_probs

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

    def select(self, root: Node) -> Node:
        """
        Select a leaf node using PUCT
        """
        node = root
        while node.children:
            node = max(node.children, key=lambda x: self.PUCT(x))
        return node

    def expand(self, node: Node, X: Dict[str, np.ndarray], y: np.ndarray) -> Node:
        """
        Expand node with NDFormer-guided action selection
        """
        valid_actions = list(self.iter_valid_action(node, shuffle=False))

        if not valid_actions:
            return node

        # Get policy prior from NDFormer
        policy_priors = self._get_policy_prior(node, valid_actions)

        # Select top-k actions based on NDFormer prior
        sorted_actions = sorted(valid_actions, key=lambda a: policy_priors[a], reverse=True)
        valid_actions = sorted_actions[:self.ndformer_topk]

        # Create child nodes
        for idx, action in enumerate(valid_actions):
            child = self.action(node, action)
            child.parent = node
            child.xchild = len(node.children)
            child.policy_prior = policy_priors[action]
            node.children.append(child)
            if self.child_num and idx + 1 >= self.child_num:
                break

        if not node.children:
            return node

        # Select child based on PUCT for simulation
        return max(node.children, key=self.PUCT)

    def fit(
        self,
        X: np.ndarray | pd.DataFrame | Dict[str, np.ndarray],
        y: np.ndarray | pd.Series,
    ):
        """
        Fit the model using NDFormer-guided MCTS

        First encodes the graph data and caches memory, then runs MCTS search
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

        # Encode graph and cache memory before MCTS search
        self._encode_and_cache_memory(X, y)

        # Call parent fit
        super().fit(X, y)
