import numbers
import warnings
import functools
import traceback
import numpy as np
from typing import List, Tuple
from ..symbols import *
from ..base_visitor import Visitor, yield_nothing


# Decorator to unpack operands for operations
# This allows us to handle operations with multiple operands in a clean way
# We can also use this decorator to suppress numpy errors
def unpack_operands():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, node, *args, **kwargs):
            # Calculate the values of the operands
            yield from yield_nothing()
            X = []
            for op in node.operands:
                x = yield (op, args, kwargs)
                X.append(x)
            # Use the defined 'visit_<Operation>' as 'func' to process the operands
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return func(self, node, *X, *args, **kwargs)

        return wrapper

    return decorator


class NumpyCalc(Visitor):
    def __call__(
        self,
        node: Symbol,
        vars: dict = {},
        edge_list: Tuple[List[int], List[int]] = None,
        num_nodes: int = None,
        use_eps: float = 0.0,
    ):
        """
        Args:
        - vars: a dictionary of variable names and their values
        - edge_list: edges (i,j) in the graph
            i and j are the indices of the nodes (starting from 0)
        - num_nodes: the number of nodes in the graph
            if not provided, it will be inferred from edge_list
        - use_eps: a small value to avoid division by zero
        """
        if num_nodes is None and edge_list is not None:
            nodes = np.unique(np.array(edge_list).reshape(-1))
            num_nodes = max(nodes) + 1

        return super().__call__(
            node,
            vars=vars,
            edge_list=edge_list,
            num_nodes=num_nodes,
            use_eps=use_eps,
        )

    def generic_visit(self, node: Symbol, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__}.visit_{type(node).__name__} not implemented"
        )

    def visit_Empty(self, node: Empty, *args, **kwargs):
        raise ValueError(
            f"Incomplete expression with Empty node is not allowed to evaluate: {node}"
        )

    def visit_Number(self, node: Number, *args, **kwargs):
        yield from yield_nothing()
        return np.asarray(node.value)

    def visit_Variable(self, node: Variable, *args, **kwargs):
        yield from yield_nothing()
        return np.asarray(kwargs["vars"][node.name])

    @unpack_operands()
    def visit_Add(self, node: Add, x1, x2, *args, **kwargs):
        return x1 + x2

    @unpack_operands()
    def visit_Sub(self, node: Sub, x1, x2, *args, **kwargs):
        return x1 - x2

    @unpack_operands()
    def visit_Mul(self, node: Mul, x1, x2, *args, **kwargs):
        return x1 * x2

    @unpack_operands()
    def visit_Div(self, node: Div, x1, x2, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return x1 / (x2 + eps * (x2 == 0))

    @unpack_operands()
    def visit_Pow(self, node: Pow, x1, x2, *args, **kwargs):
        return x1**x2

    @unpack_operands()
    def visit_Max(self, node: Max, x1, x2, *args, **kwargs):
        return np.maximum(x1, x2)

    @unpack_operands()
    def visit_Min(self, node: Min, x1, x2, *args, **kwargs):
        return np.minimum(x1, x2)

    @unpack_operands()
    def visit_Sin(self, node: Sin, x, *args, **kwargs):
        return np.sin(x)

    @unpack_operands()
    def visit_Cos(self, node: Cos, x, *args, **kwargs):
        return np.cos(x)

    @unpack_operands()
    def visit_Tan(self, node: Tan, x, *args, **kwargs):
        return np.tan(x)

    @unpack_operands()
    def visit_Sec(self, node: Sec, x, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return 1 / (np.cos(x) + eps * (np.cos(x) == 0))

    @unpack_operands()
    def visit_Csc(self, node: Csc, x, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return 1 / (np.sin(x) + eps * (np.sin(x) == 0))

    @unpack_operands()
    def visit_Cot(self, node: Cot, x, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return 1 / (np.tan(x) + eps * (np.tan(x) == 0))

    @unpack_operands()
    def visit_Log(self, node: Log, x, *args, **kwargs):
        return np.log(x)

    @unpack_operands()
    def visit_LogAbs(self, node: LogAbs, x, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return np.log(np.abs(x + eps * (x == 0)))

    @unpack_operands()
    def visit_Exp(self, node: Exp, x, *args, **kwargs):
        return np.exp(x)

    @unpack_operands()
    def visit_Abs(self, node: Abs, x, *args, **kwargs):
        return np.abs(x)

    @unpack_operands()
    def visit_Neg(self, node: Neg, x, *args, **kwargs):
        return -x

    @unpack_operands()
    def visit_Inv(self, node: Inv, x, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return 1 / (x + eps * (x == 0))

    @unpack_operands()
    def visit_Sqrt(self, node: Sqrt, x, *args, **kwargs):
        return np.sqrt(x)

    @unpack_operands()
    def visit_SqrtAbs(self, node: SqrtAbs, x, *args, **kwargs):
        return np.sqrt(np.abs(x))

    @unpack_operands()
    def visit_Pow2(self, node: Pow2, x, *args, **kwargs):
        return x**2

    @unpack_operands()
    def visit_Pow3(self, node: Pow3, x, *args, **kwargs):
        return x**3

    @unpack_operands()
    def visit_Arcsin(self, node: Arcsin, x, *args, **kwargs):
        return np.arcsin(x)

    @unpack_operands()
    def visit_Arccos(self, node: Arccos, x, *args, **kwargs):
        return np.arccos(x)

    @unpack_operands()
    def visit_Arctan(self, node: Arctan, x, *args, **kwargs):
        return np.arctan(x)

    @unpack_operands()
    def visit_Sinh(self, node: Sinh, x, *args, **kwargs):
        return np.sinh(x)

    @unpack_operands()
    def visit_Cosh(self, node: Cosh, x, *args, **kwargs):
        return np.cosh(x)

    @unpack_operands()
    def visit_Tanh(self, node: Tanh, x, *args, **kwargs):
        return np.tanh(x)

    @unpack_operands()
    def visit_Sech(self, node: Sech, x, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return 1 / (np.cosh(x) + eps * (np.cosh(x) == 0))

    @unpack_operands()
    def visit_Csch(self, node: Csch, x, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return 1 / (np.sinh(x) + eps * (np.sinh(x) == 0))

    @unpack_operands()
    def visit_Coth(self, node: Coth, x, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return 1 / (np.tanh(x) + eps * (np.tanh(x) == 0))

    @unpack_operands()
    def visit_Sigmoid(self, node: Sigmoid, x, *args, **kwargs):
        return 1 / (1 + np.exp(-x))

    @unpack_operands()
    def visit_Regular(self, node: Regular, x1, x2, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return 1 / (1 + (np.abs(x1) + eps * (x1 == 0)) ** (-x2))

    @unpack_operands()
    def visit_Sour(self, node: Sour, x, *args, **kwargs):
        """(*, n_nodes or 1) -> (*, n_edges or 1)"""
        edge_list = kwargs.get("edge_list", ([], []))

        if isinstance(x, numbers.Number) or x.size == 1:
            return x  # (1,) -> (1,)
        elif node.operands[0].nettype == "scalar" or x.shape[-1] == 1:
            if x.shape[-1] != 1:
                x = x[..., np.newaxis]
            return x  # (*, 1) -> (*, 1)
        else:
            return x[..., edge_list[0]]  # (*, V) -> (*, E)

    @unpack_operands()
    def visit_Targ(self, node: Targ, x, *args, **kwargs):
        """(*, n_nodes or 1) -> (*, n_edges or 1)"""
        edge_list = kwargs.get("edge_list", ([], []))

        if isinstance(x, numbers.Number) or x.size == 1:
            return x  # (1,) -> (1,)
        elif node.operands[0].nettype == "scalar" or x.shape[-1] == 1:
            if x.shape[-1] != 1:
                x = x[..., np.newaxis]
            return x  # (*, 1) -> (*, 1)
        else:
            return x[..., edge_list[1]]  # (*, V) -> (*, E)

    @unpack_operands()
    def visit_Aggr(self, node: Aggr, x, *args, **kwargs):
        """(*, n_edges or 1) -> (*, n_nodes)"""
        edge_list = kwargs.get("edge_list", ([], []))
        num_nodes = kwargs.get("num_nodes")

        if isinstance(x, numbers.Number) or x.size == 1:
            y = np.zeros((num_nodes,))
            np.add.at(y, edge_list[1], x)
            return y
        elif node.operands[0].nettype == "scalar" or x.shape[-1] == 1:
            if x.shape[-1] != 1:
                x = x[..., np.newaxis]
            y = np.zeros((num_nodes,))
            np.add.at(y, edge_list[1], 1)
            y = y * x
            return y
        else:
            y = np.zeros((*x.shape[:-1], num_nodes))
            for k, j in enumerate(edge_list[1]):
                y[..., j] += x[..., k]
            return y

    @unpack_operands()
    def visit_Rgga(self, node: Rgga, x, *args, **kwargs):
        """(*, n_edges or 1) -> (*, n_nodes)"""
        edge_list = kwargs.get("edge_list", ([], []))
        num_nodes = kwargs.get("num_nodes")

        if isinstance(x, numbers.Number) or x.size == 1:
            y = np.zeros((num_nodes,))
            np.add.at(y, edge_list[1], x)
            return y
        elif node.operands[0].nettype == "scalar" or x.shape[-1] == 1:
            if x.shape[-1] != 1:
                x = x[..., np.newaxis]
            y = np.zeros((num_nodes,))
            np.add.at(y, edge_list[0], 1)
            y = y * x
            return y
        else:
            y = np.zeros((*x.shape[:-1], num_nodes))
            for k, i in enumerate(edge_list[0]):
                y[..., i] += x[..., k]
            return y

    @unpack_operands()
    def visit_Readout(self, node: Readout, x, *args, **kwargs):
        """(*, n_nodes or n_edges or 1) -> (*, 1)"""
        return np.sum(x, axis=-1, keepdims=True)


"""
# 比较 aggr 不同实现方式的性能

import numpy as np

T = 100
V = 10
E = 10
x = np.random.rand(T, E)
index = np.random.randint(0, V, size=(E,))

def aggr0(x, index, V):
    y = np.zeros((*x.shape[:-1], V), dtype=x.dtype)
    for k, j in enumerate(index):
        y[..., j] += x[..., k]
    return y
    

def aggr1(x, index, V):
    x = np.asarray(x)
    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])  # shape: (B, N), B = product of leading dims

    T = x_flat.shape[0]
    y = np.zeros((T, V), dtype=x.dtype)
    for i in range(T):
        np.add.at(y[i, :], index, x_flat[i, :])
    y = y.reshape(*orig_shape[:-1], V)
    return y

def aggr2(x, index, V):
    x = np.asarray(x)
    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])  # shape: (B, N), B = product of leading dims

    T = x_flat.shape[0]
    y = np.zeros((T, V), dtype=x.dtype)
    for i in range(T):
        y[i, :] = np.bincount(index, weights=x[i, :], minlength=V)
    y = y.reshape(*orig_shape[:-1], V)
    return y

def aggr3(x, index, V):
    x = np.asarray(x)
    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])  # shape: (B, E)

    T = x_flat.shape[0]
    # 扩展 index 为 shape (T, E) 以对齐每个 batch
    index_broadcast = np.broadcast_to(index, (T, E))

    # 准备目标数组
    y_flat = np.zeros((T, V), dtype=x.dtype)

    # 扁平化批次索引：行号 (0,0,...,1,1,1,...T-1) 与 index 构成 2D 索引
    row_idx = np.repeat(np.arange(T), E)
    col_idx = index_broadcast.ravel()
    values = x_flat.ravel()

    # 聚合加和
    np.add.at(y_flat, (row_idx, col_idx), values)

    # reshape 回原来的批次形状
    return y_flat.reshape(*orig_shape[:-1], V)


y0 = aggr0(x, index, V)

y1 = aggr1(x, index, V)
assert (y0 == y1).all()

y2 = aggr2(x, index, V)
assert (y0 == y2).all()

y3 = aggr3(x, index, V)
assert (y0 == y3).all()
"""
