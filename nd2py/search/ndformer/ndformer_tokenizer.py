# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import json
import warnings
import numpy as np
from typing import List, Dict, Tuple, Optional, Literal
from ... import core as nd
from .ndformer_config import NDFormerConfig


class NumberTokenizer:
    def __init__(self, n_mantissa=4, min_exponent=-100, max_exponent=100):
        self.n_mantissa = n_mantissa
        self.min_exponent = min_exponent
        self.max_exponent = max_exponent
        self.MAX_VALUE = (10 - 10 / 10 ** n_mantissa) * 10 ** max_exponent
        self.EPS_VALUE = 10 / 10 ** n_mantissa * 10 ** min_exponent
        self.vocab = ['+', '-']
        self.vocab.extend('N' + str(x).zfill(n_mantissa) for x in range(10**n_mantissa))
        self.vocab.extend(f'E{x:+03d}' for x in range(min_exponent, max_exponent+1))
        # -9.999e100 ~ -0.001e-100 ~ -0.0000 ~ +0.0000 ~ +0.001e-100 ~ +9.999e100

    def _split_float(self, data:np.ndarray):
        """
        (+|-)(N000~N999)(E-10~E10)
            1.234e5 -> (+, N123, E5)
            -1.234e-5 -> (-, N123, E-5)
            (+-)inf -> (+|-, N999, E10)
            (+-)eps -> (+|-, N000, E-10)
            zero -> (+, N000, E-10)
            nan -> (+, N000, E10)
        """
        nan = np.isnan(data)
        data[nan] = 0.0

        sign = (data < 0).astype(int)
        data = np.abs(data)

        data[data == 0] = np.finfo(float).eps

        exponent = np.floor(np.log10(data)).clip(self.min_exponent, self.max_exponent)
        exponent[nan] = self.max_exponent
        data /= 10**exponent
        exponent = exponent.astype(int)

        mantissa = np.round(data * 10**(self.n_mantissa-1)).clip(0, 10**self.n_mantissa-1).astype(int)
        return sign, mantissa, exponent

    def _merge_float(self, data:Tuple[str, str, str]):
        sign_str, mantissa_str, exponent_str = data
        sign = -1 if sign_str == '-' else 1
        mantissa = int(mantissa_str.removeprefix('N')) / 10**(self.n_mantissa-1)
        exponent = int(exponent_str.removeprefix('E'))
        return sign * mantissa * (10 ** exponent)

    def encode(self, value: float | List[float], mode:Literal['token', 'token_id']='token') -> List[str|int]:
        if isinstance(value, float):
            value = [value]
        value = np.clip(value, -self.MAX_VALUE, self.MAX_VALUE)
        sign, mantissa, exponent = self._split_float(value)
        sign_tokens = np.where(sign, '-', '+')
        mantissa_tokens = np.array(['N' + str(x).zfill(self.n_mantissa) for x in mantissa])
        exponent_tokens = np.array([f'E{x:+03d}' for x in exponent])
        tokens = np.stack([sign_tokens, mantissa_tokens, exponent_tokens], axis=-1).reshape(-1).tolist()
        if mode == 'token_id':
            raise NotImplementedError("Token ID encoding not implemented yet.")
        return tokens
    
    def decode(self, tokens: List[str], mode:Literal['token', 'token_id']='token') -> List[float]:
        if len(tokens) % 3 != 0:
            raise ValueError("Token list length must be a multiple of 3.")
        if mode == 'token_id':
            raise NotImplementedError("Token ID decoding not implemented yet.")
        values = []
        for i in range(0, len(tokens), 3):
            value = self._merge_float((tokens[i], tokens[i+1], tokens[i+2]))
            values.append(value)
        return values


class NDFormerTokenizer:
    def __init__(self, config: NDFormerConfig, variables: Optional[List[nd.Symbol]] = None):
        self.config = config

        # Special Tokens        
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'

        # Variable Tokens
        self.variables = (
            [nd.Variable(f'n{i}', nettype='node') for i in range(1, 1+config.max_var_num)] +
            [nd.Variable(f'e{i}', nettype='edge') for i in range(1, 1+config.max_var_num)]
        ) if variables is None else variables
        self.node_var_tokens = [f'NODEVAR_{i}' for i in range(1, 1+config.max_var_num)]
        self.edge_var_tokens = [f'EDGEVAR_{i}' for i in range(1, 1+config.max_var_num)]
        self.scalar_var_tokens = [f'SCALARVAR_{i}' for i in range(1, 1+config.max_var_num)]
        self.variable_mapping = {}
        for idx, var in enumerate([var for var in self.variables if var.nettype == 'node']):
            self.variable_mapping[var.name] = self.node_var_tokens[idx]
        for idx, var in enumerate([var for var in self.variables if var.nettype == 'edge']):
            self.variable_mapping[var.name] = self.edge_var_tokens[idx]
        for idx, var in enumerate([var for var in self.variables if var.nettype == 'scalar']):
            self.variable_mapping[var.name] = self.scalar_var_tokens[idx]
        self.i_variable_mapping = {v: k for k, v in self.variable_mapping.items()}

        # Number Tokens
        self.num_tokenizer = NumberTokenizer(
            n_mantissa=config.n_mantissa, 
            min_exponent=config.min_exponent, 
            max_exponent=config.max_exponent
        )

        # Parent Tokens
        self.parent_tokens = [f'INDEX-{i}' for i in range(config.max_seq_len)] + ['INDEX-ROOT']

        # Nettype Tokens
        self.nettype_tokens = [f'NETTYPE-{nt.upper()}' for nt in ['NODE', 'EDGE', 'SCALAR', 'NONE']]

        # Empty Token
        self.empty_token = 'EMPTY'

        # All Tokens
        self.vocab = [
            self.pad_token, self.sos_token, self.eos_token, self.unk_token,
            *self.node_var_tokens,
            *self.edge_var_tokens,
            *self.scalar_var_tokens,
            *self.config.operands,
            *self.num_tokenizer.vocab,
            *self.parent_tokens,
            *self.nettype_tokens,
            self.empty_token,
        ]
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
    
    @property
    def vocab_size(self): return len(self.vocab)
    @property
    def pad_token_id(self): return self.token2id[self.pad_token]
    @property
    def sos_token_id(self): return self.token2id[self.sos_token]
    @property
    def eos_token_id(self): return self.token2id[self.eos_token]
    @property
    def unk_token_id(self): return self.token2id[self.unk_token]

    def encode(self, eqtree: nd.Symbol, mode:Literal['token', 'token_id']='token') -> Tuple[List[int], List[int], List[int]]:
        tokens = []
        parents = []
        nettypes = []
        symbol_list = list(eqtree.iter_preorder())
        for symbol in symbol_list:
            if isinstance(symbol, nd.Empty):
                tokens.append(self.empty_token)
            elif isinstance(symbol, nd.Variable):
                tokens.append(self.variable_mapping[symbol.name])
            elif isinstance(symbol, nd.Number):
                tokens.extend(self.num_tokenizer.encode(symbol.value, mode='token'))
            else:
                tokens.append(type(symbol).__name__)
            parents.append("INDEX-" + (str(symbol_list.index(symbol.parent)) if symbol.parent is not None else "ROOT"))
            nettypes.append("NETTYPE-" + (symbol.nettype or 'NONE').upper())

        if any(unknown_tokens := [token not in self.token2id for token in tokens + parents + nettypes]):
            warnings.warn(f"Found unknown tokens during encoding: {set(unknown_tokens)}. They will be replaced with <UNK>.")
            tokens = [token if token in self.token2id else self.unk_token for token in tokens]
            parents = [token if token in self.token2id else self.unk_token for token in parents]
            nettypes = [token if token in self.token2id else self.unk_token for token in nettypes]

        if mode == 'token_id':
            token_ids = [self.token2id[token] for token in tokens]
            parent_ids = [self.token2id[token] for token in parents]
            nettype_ids = [self.token2id[token] for token in nettypes]
            return token_ids, parent_ids, nettype_ids
        else:
            return tokens, parents, nettypes
    
    def decode(self, tokens: List[str], parents: List[str], nettypes: List[str], mode:Literal['token', 'token_id']='token') -> nd.Symbol:
        if mode == 'token_id':
            tokens = [self.id2token[token_id] for token_id in tokens]

        preorder = []
        for token in (tokens := iter(tokens)):
            if token in [self.sos_token, self.pad_token, self.eos_token]:
                continue
            if token in ['+', '-']:
                symbol = nd.Number(self.num_tokenizer.decode([token, next(tokens), next(tokens)], mode='token')[0])
            elif token in self.i_variable_mapping:
                symbol = nd.Variable(self.i_variable_mapping[token])
            elif token in self.config.operands:
                symbol = getattr(nd, token)()
            elif token == self.empty_token:
                symbol = nd.Empty()
            else:
                raise ValueError(f"Unknown token during decoding: {token}")
            preorder.append(symbol)
        eqtree = nd.from_preorder(preorder)
        return eqtree

    def encode_array(self, data: np.ndarray, mode: Literal['token', 'token_id'] = 'token_id'):
        """专门用于将纯浮点数组转换为 token 或 token_id"""
        shape = data.shape
        tokens = self.num_tokenizer.encode(data.astype(float).reshape(-1)) 
        if mode == 'token_id':
            token_ids = [self.token2id[token] for token in tokens]
            return np.array(token_ids).reshape(*shape, 3)
        return np.array(tokens).reshape(*shape, 3)
    
    def decode_array(self, tokens: np.ndarray, mode: Literal['token', 'token_id'] = 'token_id'):
        """专门用于将 token 或 token_id 数组转换回纯浮点数组"""
        if mode == 'token_id':
            tokens = np.vectorize(lambda x: self.id2token.get(x, self.unk_token))(tokens)
        flat_tokens = tokens.reshape(-1, 3)
        values = [self.num_tokenizer.decode(token_list, mode='token')[0] for token_list in flat_tokens]
        return np.array(values).reshape(tokens.shape[:-1])

    def to_dict(self) -> dict:
        """导出核心配置以供序列化"""
        return {
            "operators": self.operators,
            "max_dim_node": self.max_dim_node,
            "max_dim_edge": self.max_dim_edge,
            "vocab_size": len(self.vocab)
        }

    @classmethod
    def from_dict(cls, config: dict) -> 'NDFormerTokenizer':
        return cls(config["operators"], config["max_dim_node"], config["max_dim_edge"])

    def save(self, filepath: str):
        """保存到本地 JSON 文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> 'NDFormerTokenizer':
        """从本地 JSON 文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return cls.from_dict(config)

    def __eq__(self, other) -> bool:
        """
        重写等于运算符。
        用法: if tokenizer == cached_tokenizer: print("配置一致！")
        """
        if not isinstance(other, NDFormerTokenizer):
            return False
        return self.to_dict() == other.to_dict()
