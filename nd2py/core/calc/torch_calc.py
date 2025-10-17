import torch
import numbers
import functools
import numpy as np
from ..symbols import *
from typing import List, Tuple
from ..base_visitor import Visitor, yield_nothing


# Decorator to unpack operands for operations
# This allows us to handle operations with multiple operands in a clean way
# We can also use this decorator to replace NaN values in the input and output
def unpack_operands(
    mask_out_nan=False,
    double_check_nan=False,
    fill_nan_input=1.0,
    fill_nan_output=torch.nan,
):
    """Decorator to unpack operands of a node and apply a function to them.
    Args:
    - mask_out_nan: whether to replace NaN values in the input with fill_nan_input
    - double_check_nan: whether to calculate the output for invalid inputs
        Set to True can lead to performance degradation, but helps with operations like Div and Inv which map Non-nan to nan.
    - fill_nan_input: value to replace NaN values in the input
        This can be any value as long as 'func' does not map it to nan.
    - fill_nan_output: value to replace NaN values in the output
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, node, *args, **kwargs):
            yield from yield_nothing()
            X = []
            for op in node.operands:
                x = yield (op, args, kwargs)
                X.append(x)
            X = list(torch.broadcast_tensors(*X))
            if not mask_out_nan:
                return func(self, node, *X, *args, **kwargs)
            if double_check_nan:
                valid = func(self, node, *X, *args, **kwargs).isfinite()
            else:
                valid = torch.stack([x.isfinite() for x in X], dim=0).all(dim=0)
            for idx, x in enumerate(X):
                X[idx] = torch.where(valid, x, fill_nan_input)
            y = func(self, node, *X, *args, **kwargs)
            y = torch.where(valid, y, fill_nan_output)
            return y

        return wrapper

    return decorator


class TorchCalc(Visitor):
    def __call__(
        self,
        node: Symbol,
        vars: dict = {},
        edge_list: Tuple[List[int], List[int]] = None,
        num_nodes: int = None,
        use_eps: float = 0.0,
        device: str = "cpu",
    ):
        """
        Args:
        - vars: a dictionary of variable names and their values
        - edge_list: edges (i,j) in the graph
            i and j are the indices of the nodes (starting from 0)
        - num_nodes: the number of nodes in the graph
            if not provided, it will be inferred from edge_list
        - use_eps: a small value to avoid division by zero
        - device: cpu or cuda
        """
        if num_nodes is None and edge_list is not None:
            nodes = np.unique(np.array(edge_list).reshape(-1))
            num_nodes = max(nodes) + 1

        if edge_list is not None:
            edge_list = (
                torch.tensor(edge_list[0], device=device),
                torch.tensor(edge_list[1], device=device),
            )

        try:
            return super().__call__(
                node,
                vars=vars,
                edge_list=edge_list,
                num_nodes=num_nodes,
                use_eps=use_eps,
                device=device,
            )
        except Exception as e:
            raise ValueError(f"Error in {type(self).__name__}({node}): {e}") from e

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
        device = kwargs.get("device")
        return torch.as_tensor(node.value, device=device)

    def visit_Variable(self, node: Variable, *args, **kwargs):
        yield from yield_nothing()
        device = kwargs.get("device")
        return torch.as_tensor(kwargs["vars"][node.name], device=device)

    @unpack_operands(mask_out_nan=True)
    def visit_Add(self, node: Add, x1, x2, *args, **kwargs):
        return x1 + x2

    @unpack_operands(mask_out_nan=True)
    def visit_Sub(self, node: Sub, x1, x2, *args, **kwargs):
        return x1 - x2

    @unpack_operands(mask_out_nan=True)
    def visit_Mul(self, node: Mul, x1, x2, *args, **kwargs):
        return x1 * x2

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Div(self, node: Div, x1, x2, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return x1 / (x2 + eps * (x2 == 0))

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Pow(self, node: Pow, x1, x2, *args, **kwargs):
        return x1**x2

    @unpack_operands(mask_out_nan=True)
    def visit_Max(self, node: Max, x1, x2, *args, **kwargs):
        return torch.max(x1, x2)

    @unpack_operands(mask_out_nan=True)
    def visit_Min(self, node: Min, x1, x2, *args, **kwargs):
        return torch.min(x1, x2)

    @unpack_operands(mask_out_nan=True)
    def visit_Sin(self, node: Sin, x, *args, **kwargs):
        return torch.sin(x)

    @unpack_operands(mask_out_nan=True)
    def visit_Cos(self, node: Cos, x, *args, **kwargs):
        return torch.cos(x)

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Tan(self, node: Tan, x, *args, **kwargs):
        return torch.tan(x)

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Sec(self, node: Sec, x, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return 1 / (torch.cos(x) + eps * (torch.cos(x) == 0))

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Csc(self, node: Csc, x, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return 1 / (torch.sin(x) + eps * (torch.sin(x) == 0))

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Cot(self, node: Cot, x, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return 1 / (torch.tan(x) + eps * (torch.tan(x) == 0))

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Log(self, node: Log, x, *args, **kwargs):
        return torch.log(x)

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_LogAbs(self, node: LogAbs, x, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return torch.log(torch.abs(x + eps * (x == 0)))

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Exp(self, node: Exp, x, *args, **kwargs):
        return torch.exp(x)

    @unpack_operands(mask_out_nan=True)
    def visit_Abs(self, node: Abs, x, *args, **kwargs):
        return torch.abs(x)

    @unpack_operands(mask_out_nan=True)
    def visit_Neg(self, node: Neg, x, *args, **kwargs):
        return -x

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Inv(self, node: Inv, x, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return 1 / (x + eps * (x == 0))

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Sqrt(self, node: Sqrt, x, *args, **kwargs):
        return torch.sqrt(x)

    @unpack_operands(mask_out_nan=True)
    def visit_SqrtAbs(self, node: SqrtAbs, x, *args, **kwargs):
        return torch.sqrt(torch.abs(x))

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Pow2(self, node: Pow2, x, *args, **kwargs):
        return x**2

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Pow3(self, node: Pow3, x, *args, **kwargs):
        return x**3

    @unpack_operands(mask_out_nan=True)
    def visit_Arcsin(self, node: Arcsin, x, *args, **kwargs):
        return torch.arcsin(x)

    @unpack_operands(mask_out_nan=True)
    def visit_Arccos(self, node: Arccos, x, *args, **kwargs):
        return torch.arccos(x)

    @unpack_operands(mask_out_nan=True)
    def visit_Arctan(self, node: Arctan, x, *args, **kwargs):
        return torch.arctan(x)

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Sinh(self, node: Sinh, x, *args, **kwargs):
        return torch.sinh(x)

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Cosh(self, node: Cosh, x, *args, **kwargs):
        return torch.cosh(x)

    @unpack_operands(mask_out_nan=True)
    def visit_Tanh(self, node: Tanh, x, *args, **kwargs):
        return torch.tanh(x)

    @unpack_operands(mask_out_nan=True)
    def visit_Sech(self, node: Sech, x, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return 1 / (torch.cosh(x) + eps * (torch.cosh(x) == 0))

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Csch(self, node: Csch, x, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return 1 / (torch.sinh(x) + eps * (torch.sinh(x) == 0))

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Coth(self, node: Coth, x, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return 1 / (torch.tanh(x) + eps * (torch.tanh(x) == 0))

    @unpack_operands(mask_out_nan=True)
    def visit_Sigmoid(self, node: Sigmoid, x, *args, **kwargs):
        return torch.sigmoid(x)

    @unpack_operands(mask_out_nan=True, double_check_nan=True)
    def visit_Regular(self, node: Regular, x1, x2, *args, **kwargs):
        eps = kwargs.get("use_eps")
        return 1 / (1 + (torch.abs(x1) + eps * (x1 == 0)) ** (-x2))

    @unpack_operands()
    def visit_Sour(self, node: Sour, x, *args, **kwargs):
        """(*, n_nodes or 1) -> (*, n_edges or 1)"""
        edge_list = kwargs.get("edge_list", ([], []))

        if isinstance(x, numbers.Number) or x.size == 1:
            return x  # (1,) -> (1,)
        elif node.operands[0].nettype == "scalar":
            if x.shape[-1] != 1:
                x = x[..., None]
            return x  # (*, 1) -> (*, 1)
        else:
            return x[..., edge_list[0]]  # (*, V) -> (*, E)

    @unpack_operands()
    def visit_Targ(self, node: Targ, x, *args, **kwargs):
        """(*, n_nodes or 1) -> (*, n_edges or 1)"""
        edge_list = kwargs.get("edge_list", ([], []))

        if isinstance(x, numbers.Number) or x.size == 1:
            return x  # (1,) -> (1,)
        elif node.operands[0].nettype == "scalar":
            if x.shape[-1] != 1:
                x = x[..., None]
            return x  # (*, 1) -> (*, 1)
        else:
            return x[..., edge_list[1]]  # (*, V) -> (*, E)

    @unpack_operands()
    def visit_Aggr(self, node: Aggr, x, *args, **kwargs):
        """(*, n_edges or 1) -> (*, n_nodes)"""
        edge_list = kwargs.get("edge_list", ([], []))
        num_nodes = kwargs.get("num_nodes")
        device = kwargs.get("device")

        if isinstance(x, numbers.Number) or x.size == 1:
            y = torch.zeros((num_nodes,), dtype=x.dtype, device=device)
            y = torch.scatter_add(y, dim=-1, index=edge_list[1].expand_as(x), src=x)
            return y
        elif node.operands[0].nettype == "scalar":
            if x.shape[-1] != 1:
                x = x[..., None]
            y = torch.zeros((num_nodes,), dtype=x.dtype, device=device)
            y = torch.scatter_add(y, dim=-1, index=edge_list[1].expand_as(x), src=1)
            y = y * x
            return y
        else:
            y = torch.zeros((*x.shape[:-1], num_nodes), dtype=x.dtype, device=device)
            y = torch.scatter_add(y, dim=-1, index=edge_list[1].expand_as(x), src=x)
            return y

    @unpack_operands()
    def visit_Rgga(self, node: Rgga, x, *args, **kwargs):
        """(*, n_edges or 1) -> (*, n_nodes)"""
        edge_list = kwargs.get("edge_list", ([], []))
        num_nodes = kwargs.get("num_nodes")
        device = kwargs.get("device")

        if isinstance(x, numbers.Number) or x.size == 1:
            y = torch.zeros((num_nodes,), dtype=x.dtype, device=device)
            y = torch.scatter_add(y, dim=-1, index=edge_list[0].expand_as(x), src=x)
            return y
        elif node.operands[0].nettype == "scalar":
            if x.shape[-1] != 1:
                x = x[..., None]
            y = torch.zeros((num_nodes,), dtype=x.dtype, device=device)
            y = torch.scatter_add(y, dim=-1, index=edge_list[0].expand_as(x), src=1)
            y = y * x
            return y
        else:
            y = torch.zeros((*x.shape[:-1], num_nodes), dtype=x.dtype, device=device)
            y = torch.scatter_add(y, dim=-1, index=edge_list[0].expand_as(x), src=x)
            return y

    @unpack_operands()
    def visit_Readout(self, node: Readout, x, *args, **kwargs):
        """(*, n_nodes or n_edges or 1) -> (*, 1)"""
        return torch.sum(x, axis=-1, keepdim=True)
