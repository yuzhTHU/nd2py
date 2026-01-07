import warnings
from typing import List, Literal, Tuple, Any
from ..context.warn_once import warn_once
from ..tree.tree_mixin import TreeMixin
from ..nettype.nettype_mixin import NetTypeMixin, NetType

__all__ = ["Symbol"]


def is_number(value, scalar_only=False):
    """Check if the value is a number or a numpy array"""
    import numbers
    import numpy as np
    from .number import Number

    if scalar_only:
        if isinstance(value, (numbers.Number, np.number)):
            return True
        elif isinstance(value, np.ndarray) and value.ndim == 0:
            return value.dtype.kind in "biufc"
        elif isinstance(value, Number) and value.nettype == "scalar":
            return True
        else:
            return False
    else:
        if isinstance(value, (numbers.Number, np.number)):
            return True
        elif isinstance(value, np.ndarray):
            return value.dtype.kind in "biufc"
        elif isinstance(value, Number):
            return True
        elif isinstance(value, (list, tuple, set)):
            return all(is_number(v, scalar_only=scalar_only) for v in value)


def is_integer_number(x):
    import math
    import numbers
    import numpy as np

    # 排除 NaN 和 Inf
    if isinstance(x, numbers.Number):
        if math.isinf(x) or math.isnan(x):
            return False
        # 检查是否是整数（包括 float 表示的整数）
        return float(x).is_integer() and np.abs(x) < 10
    return False


class SymbolMeta(type):
    def __repr__(cls):
        return cls.__name__


class Symbol(NetTypeMixin, TreeMixin, metaclass=SymbolMeta):
    n_operands = None

    def __init__(self, *operands, nettype: NetType = None):
        NetTypeMixin.__init__(self, fixed_nettype=nettype)

        # 连接父代 (这里自底向上创建符号树, 没有 parent 需要连接)
        self.parent = None

        # 预处理并连接子代
        self.operands = self._sanitize_operands(operands)
        for child in self.operands:
            child.parent = self

        # 让 operands 在 self 的框架下相互约束直至稳定
        for child in self.operands:
            child.set_parent_with_constraint(self)

        # 根据子节点的状态收缩自己的 candidates
        if not self.propagate():
            raise ValueError(
                "NetType Conflict: Cannot determine nettype for this Symbol based on its operands."
            )

    def _sanitize_operands(self, operands: List["Symbol"]) -> List["Symbol"]:
        """将输入的 operands 转换为标准的 Symbol 列表"""
        from .empty import Empty
        from .number import Number
        from .variable import Variable

        operands = list(operands)

        # 如果 operands 数量不对，尝试用 Empty 补全
        if self.n_operands == len(operands):
            pass
        elif len(operands) == 0:
            operands = [Empty() for _ in range(self.n_operands)]
        else:
            raise ValueError(
                f"Expected {self.n_operands} operands in {self.__class__.__name__}, "
                f"got {len(operands)}"
            )

        # 类型转换与检查
        for i, op in enumerate(operands):
            if isinstance(op, Symbol):
                pass
            elif is_number(op, scalar_only=True):
                operands[i] = Number(op, nettype="scalar")
            else:
                raise TypeError(
                    f"Invalid operand type: {type(op)}, expected Symbol or number."
                )

        # 所有权检查以保持树形结构
        for i, child in enumerate(operands):
            if child.parent is not None:
                if not isinstance(child, Variable) and warn_once(
                    "subexpression_with_parent"
                ):
                    warnings.warn(
                        f"The object '{child}' cannot serve as a subexpression in multiple locations. "
                        f"It will be copied to avoid this behavior.",
                        category=UserWarning,
                        stacklevel=2,
                    )
                operands[i] = child.copy()
        return operands

    def __repr__(self):
        return self.to_str()

    def __str__(self):
        return self.to_str()

    def __len__(self):
        from ..basic.get_length import GetLength

        return GetLength()(self)

    def __add__(self, other):
        from .operands import Add

        return Add(self, other)

    def __radd__(self, other):
        from .operands import Add

        return Add(other, self)

    def __sub__(self, other):
        from .operands import Sub

        return Sub(self, other)

    def __rsub__(self, other):
        from .operands import Sub

        return Sub(other, self)

    def __mul__(self, other):
        from .operands import Mul

        return Mul(self, other)

    def __rmul__(self, other):
        from .operands import Mul

        return Mul(other, self)

    def __truediv__(self, other):
        from .operands import Div

        return Div(self, other)

    def __rtruediv__(self, other):
        from .operands import Div

        return Div(other, self)

    def __pow__(self, other):
        from .operands import Pow2, Pow3, Sqrt, Inv, Pow

        if other == 2.0:
            return Pow2(self)
        if other == 3.0:
            return Pow3(self)
        if other == 0.5:
            return Sqrt(self)
        if other == -1.0:
            return Inv(self)
        return Pow(self, other)

    def __rpow__(self, other):
        from .operands import Pow

        return Pow(other, self)

    def __neg__(self):
        from .operands import Neg

        return Neg(self)

    def __invert__(self):
        from .operands import Inv

        return Inv(self)

    def copy(self):
        """Create a deep copy of the Symbol. The result will not inherit self.parent"""
        from ..basic.get_copy import GetCopy

        return GetCopy()(self)

    def get_numbers(
        self,
        fitable_only: bool = False,
        float_only: bool = False,
        scalar_only: bool = False,
    ) -> List["Number"]:
        """Get the Numbers in the Symbol.
        Args:
        - fitable_only: bool, whether to return only the fitable Numbers
        - float_only: bool, whether to return only the float Numbers
        Returns:
        - List[Number], a list of Numbers
        """
        numbers = []
        for op in self.iter_preorder():
            if (
                is_number(op, scalar_only=scalar_only)
                and (not fitable_only or op.fitable)
                and (not float_only or not is_integer_number(op.value))
            ):
                numbers.append(op)
        return numbers

    def get_parameters(
        self, fitable_only: bool = False, float_only: bool = False
    ) -> List[float]:
        """Get the parameters of the Symbol.
        Args:
        - fitable_only: bool, whether to return only the fitable parameters
        - float_only: bool, whether to return only the float parameters
        Returns:
        - List[float], a list of parameters
        """
        params = []
        for op in self.iter_preorder():
            if (
                is_number(op)
                and (not fitable_only or op.fitable)
                and (not float_only or not is_integer_number(op.value))
            ):
                params.append(op.value)
        return params

    def set_parameters(
        self, params: List[float], fitable_only: bool = False, float_only: bool = False
    ):
        """Set the parameters of the Symbol.
        Args:
        - params: List[float], a list of parameters
        - fitable_only: bool, whether to set only the fitable parameters
        - float_only: bool, whether to set only the float parameters
        """
        if len(params) != len(
            self.get_parameters(fitable_only=fitable_only, float_only=float_only)
        ):
            raise ValueError(
                f"params length {len(params)} does not match the number of parameters {len(self.get_parameters(fitable_only=fitable_only, float_only=float_only))} "
            )

        param_iter = iter(params)
        for op in self.iter_preorder():
            if (
                is_number(op)
                and (not fitable_only or op.fitable)
                and (not float_only or not is_integer_number(op.value))
            ):
                op.value = next(param_iter)

    def to_str(
        self,
        raw=False,
        latex=False,
        number_format="",
        omit_mul_sign=False,
        skeleton=False,
    ) -> str:
        """
        Args:
        - raw:bool=False, whether to return the raw format
        - number_format:str='', can be '0.2f'
        - omit_mul_sign:bool=False, whether to omit the multiplication sign
        - latex:bool=False, whether to return the latex format
        - skeleton:bool=False, whether to ignore the concrete values of Number
        """
        from ..printer.string_printer import StringPrinter

        return StringPrinter()(
            self,
            raw=raw,
            latex=latex,
            number_format=number_format,
            omit_mul_sign=omit_mul_sign,
            skeleton=skeleton,
        )

    def to_tree(self, number_format="", flat=False, skeleton=False) -> str:
        """
        Args:
        - number_format:str='', can be '0.2f'
        - flat:bool=False, whether to flat the Add and Mul
        - omit_mul_sign:bool=False, whether to omit the multiplication sign
        """
        from ..printer.tree_printer import TreePrinter

        return TreePrinter()(
            self, number_format=number_format, flat=flat, skeleton=skeleton
        )

    def eval(
        self,
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
        from ..calc.numpy_calc import NumpyCalc

        return NumpyCalc()(
            self, vars=vars, edge_list=edge_list, num_nodes=num_nodes, use_eps=use_eps
        )

    def eval_torch(
        self,
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
        from ..calc.torch_calc import TorchCalc

        return TorchCalc()(
            self,
            vars=vars,
            edge_list=edge_list,
            num_nodes=num_nodes,
            use_eps=use_eps,
            device=device,
        )

    def split_by_add(
        self,
        split_by_sub: bool = False,
        expand_mul: bool = False,
        expand_div: bool = False,
        expand_aggr: bool = False,
        expand_rgga: bool = False,
        expand_sour: bool = False,
        expand_targ: bool = False,
        expand_readout: bool = False,
        remove_coefficients: bool = False,
        merge_bias: bool = False,
    ) -> List["Symbol"]:
        """Split the node by addition, returning a list of symbols.
        Args:
        - node: Symbol, the node to split
        - split_by_sub: bool, whether to split by Sub nodes
        - expand_mul: bool, whether to expand Mul nodes
        - expand_div: bool, whether to expand Div nodes
        - expand_aggr: bool, whether to expand Aggr nodes
        - expand_rgga: bool, whether to expand Rgga nodes
        - expand_sour: bool, whether to expand Sour nodes
        - expand_targ: bool, whether to expand Targ nodes
        - expand_readout: bool, whether to expand Readout (Readout(a + b) -> [Readout(a), Readout(b)])
        - remove_coefficients: bool, whether to remove coefficients from the symbols
        - merge_bias: bool, whether to merge bias terms
        """
        from ..transform.split_by_add import SplitByAdd

        return SplitByAdd()(
            self,
            split_by_sub=split_by_sub,
            expand_mul=expand_mul,
            expand_div=expand_div,
            expand_aggr=expand_aggr,
            expand_rgga=expand_rgga,
            expand_sour=expand_sour,
            expand_targ=expand_targ,
            expand_readout=expand_readout,
            remove_coefficients=remove_coefficients,
            merge_bias=merge_bias,
        )

    def split_by_mul(
        self,
        split_by_div: bool = False,
        merge_coefficients: bool = False,
    ) -> List["Symbol"]:
        """Split the node by multiplication, returning a list of symbols.
        Args:
        - node: Symbol, the node to split (a * b * c -> [a, b, c])
        - split_by_div: bool, whether to split by Div (a / b -> [a, b])
        - merge_coefficients: bool, whether to merge coefficients from the symbols
        """
        from ..transform.split_by_mul import SplitByMul

        return SplitByMul()(
            self,
            split_by_div=split_by_div,
            merge_coefficients=merge_coefficients,
        )

    def fix_nettype(
        self,
        nettype: NetType = "node",
        direction: Literal["bottom-up", "top-down"] = "top-down",
        edge_to_node=["remove_targ", "remove_sour", "add_aggr", "add_rgga"],
        node_to_edge=["remove_aggr", "remove_rgga", "add_targ", "add_sour"],
        edge_to_scalar=["remove_sour", "remove_targ", "add_readout"],
        node_to_scalar=["remove_aggr", "remove_rgga", "add_readout"],
        scalar_to_node=["keep"],
        scalar_to_edge=["keep"],
    ):
        """fix the nettype of symbols in an expression, useful in GP or LLMSR where equations are generated randomly and can have incorrect nettypes
        - node: the root symbol of the expression to fix
        - nettype: the nettype to set for the symbols, can be 'node', 'edge', or 'scalar'
        - direction: the direction of the fix, can be 'bottom-up' or 'top-down'
        - edge_to_node: list of operations to convert edge symbols to node symbols
        - node_to_edge: list of operations to convert node symbols to edge symbols
        - edge_to_scalar: list of operations to convert edge symbols to scalar symbols
        - node_to_scalar: list of operations to convert node symbols to scalar symbols
        - scalar_to_node: list of operations to convert scalar symbols to node symbols
        - scalar_to_edge: list of operations to convert scalar symbols to edge symbols
        """
        from ..transform.fix_nettype import FixNetType

        return FixNetType()(
            self,
            nettype=nettype,
            direction=direction,
            edge_to_node=edge_to_node,
            node_to_edge=node_to_edge,
            edge_to_scalar=edge_to_scalar,
            node_to_scalar=node_to_scalar,
            scalar_to_node=scalar_to_node,
            scalar_to_edge=scalar_to_edge,
        )

    def simplify(
        self,
        transform_constant_subtree: bool = True,
        remove_useless_readout: bool = True,
        remove_nested_sin: bool = False,
        remove_nested_cos: bool = False,
        remove_nested_tanh: bool = False,
        remove_nested_sigmoid: bool = False,
        remove_nested_sqrt: bool = False,
        remove_nested_sqrtabs: bool = False,
        remove_nested_exp: bool = False,
        remove_nested_log: bool = False,
        remove_nested_logabs: bool = False,
    ):
        from ..transform.simplify import Simplify

        return Simplify()(
            self,
            transform_constant_subtree=transform_constant_subtree,
            remove_useless_readout=remove_useless_readout,
            remove_nested_sin=remove_nested_sin,
            remove_nested_cos=remove_nested_cos,
            remove_nested_tanh=remove_nested_tanh,
            remove_nested_sigmoid=remove_nested_sigmoid,
            remove_nested_sqrt=remove_nested_sqrt,
            remove_nested_sqrtabs=remove_nested_sqrtabs,
            remove_nested_exp=remove_nested_exp,
            remove_nested_log=remove_nested_log,
            remove_nested_logabs=remove_nested_logabs,
        )
