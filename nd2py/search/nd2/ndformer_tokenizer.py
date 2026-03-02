import json
from typing import List, Dict, Tuple, Optional

class GraphEquationTokenizer:
    def __init__(self, operators: List[str], max_dim_node: int, max_dim_edge: int):
        self.max_dim_node = max_dim_node
        self.max_dim_edge = max_dim_edge
        self.operators = operators # 保存以供序列化
        
        # 1. 定义特殊 Token
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'
        
        # 2. 构建基础词表
        self.vocab = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        self.vocab.extend(operators)
        
        # 3. 构建内部抽象变量 Token
        self.node_var_tokens = [f'N_{i}' for i in range(max_dim_node)]
        self.edge_var_tokens = [f'E_{i}' for i in range(max_dim_edge)]
        
        self.vocab.extend(self.node_var_tokens)
        self.vocab.extend(self.edge_var_tokens)
        
        # 4. 创建双向映射字典
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}

    # ==========================================
    # 多重遍历逻辑
    # ==========================================
    def _tree_to_preorder(self, node) -> List[str]:
        if node.is_leaf: return [str(node.value)]
        seq = [node.operator]
        for child in node.children:
            seq.extend(self._tree_to_preorder(child))
        return seq

    def _tree_to_postorder(self, node) -> List[str]:
        if node.is_leaf: return [str(node.value)]
        seq = []
        for child in node.children:
            seq.extend(self._tree_to_postorder(child))
        seq.append(node.operator)
        return seq

    def _tree_to_inorder(self, node) -> List[str]:
        # 注意：中序遍历对于非二叉树定义模糊。假设你的树绝大多数是二叉的（如 +,-,*,/）
        if node.is_leaf: return [str(node.value)]
        if len(node.children) == 1:
            # 单目运算符，如 sin(x)
            return [node.operator] + self._tree_to_inorder(node.children[0])
        elif len(node.children) == 2:
            return self._tree_to_inorder(node.children[0]) + [node.operator] + self._tree_to_inorder(node.children[1])
        else:
            # 兼容多叉节点的回退逻辑
            seq = [node.operator]
            for child in node.children: seq.extend(self._tree_to_inorder(child))
            return seq

    # ==========================================
    # 核心编码逻辑 (输出 Tuple[List, List, List])
    # ==========================================
    def encode(self, equation_tree, node_vars: List[str], edge_vars: List[str]) -> Tuple[List[int], List[int], List[int]]:
        mapping = self._get_var_mapping(node_vars, edge_vars)
        
        def _symbols_to_ids(symbols: List[str]) -> List[int]:
            return [self.token2id.get(mapping.get(sym, sym), self.token2id[self.unk_token]) for sym in symbols]

        # 分别生成并转换三种序列
        pre_ids = _symbols_to_ids(self._tree_to_preorder(equation_tree))
        in_ids = _symbols_to_ids(self._tree_to_inorder(equation_tree))
        post_ids = _symbols_to_ids(self._tree_to_postorder(equation_tree))
        
        return pre_ids, in_ids, post_ids

    def _get_var_mapping(self, node_vars: List[str], edge_vars: List[str]) -> Dict[str, str]:
        # 与之前一致，生成 {'mass': 'N_0'} 这样的映射
        mapping = {}
        for i, var in enumerate(node_vars): mapping[var] = self.node_var_tokens[i]
        for i, var in enumerate(edge_vars): mapping[var] = self.edge_var_tokens[i]
        return mapping

    # ==========================================
    # 序列化、反序列化与缓存一致性校验
    # ==========================================
    def to_dict(self) -> dict:
        """导出核心配置以供序列化"""
        return {
            "operators": self.operators,
            "max_dim_node": self.max_dim_node,
            "max_dim_edge": self.max_dim_edge,
            "vocab_size": len(self.vocab)
        }

    @classmethod
    def from_dict(cls, config: dict) -> 'GraphEquationTokenizer':
        return cls(config["operators"], config["max_dim_node"], config["max_dim_edge"])

    def save(self, filepath: str):
        """保存到本地 JSON 文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> 'GraphEquationTokenizer':
        """从本地 JSON 文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return cls.from_dict(config)

    def __eq__(self, other) -> bool:
        """
        重写等于运算符。
        用法: if tokenizer == cached_tokenizer: print("配置一致！")
        """
        if not isinstance(other, GraphEquationTokenizer):
            return False
        return self.to_dict() == other.to_dict()