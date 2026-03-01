# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import warnings
from abc import ABCMeta
from typing import List, Literal, Tuple, Any, Set, Optional
from ..context.warn_once import warn_once
from ..symbol_api import SymbolAPIMixin
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


class SymbolMeta(ABCMeta):
    def __repr__(cls):
        return cls.__name__


class Symbol(NetTypeMixin, TreeMixin, SymbolAPIMixin, metaclass=SymbolMeta):
    n_operands = None

    def __init__(self, *operands, nettype: Optional[NetType|Set[NetType]] = None):
        NetTypeMixin.__init__(self, nettype=nettype)

        # 连接父代 (这里自底向上创建符号树, 没有 parent 需要连接)
        self.parent = None

        # 处理并连接子代 (子代在 _sanitize_operands 中被连接到父代)
        self.operands = self._sanitize_operands(operands)

        # 触发全树的 nettype 更新以传播约束 (如果有的话)
        # 大规模构造符号树时可能导致重复 inference 产生性能问题, 需要结合 no_nettype_inference 上下文管理器并在最后手动调用 infer_nettype() 以优化性能
        self.infer_nettype()

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
                if not isinstance(child, Variable) and warn_once("subexpression_with_parent"):
                    warnings.warn(
                        f"The object '{child}' cannot serve as a subexpression in multiple locations. "
                        f"It will be copied to avoid this behavior.",
                        category=UserWarning,
                        stacklevel=2,
                    )
                operands[i] = child.copy()
            operands[i].parent = self
        return operands

    def __repr__(self):
        return self.to_str()

    def __str__(self):
        return self.to_str()

    def __len__(self):
        from ..basic import GetLength

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
        from ..basic import GetCopy

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

    @classmethod
    def map_nettype(cls, *children_nettypes: NetType) -> Optional[NetType]:
        """ 默认的 nettype 映射逻辑: node 和 edge 不能相互运算; 只有 scalar 时返回 scalar; 否则返回 node / edge """
        if len(children_nettypes) != cls.n_operands:
            raise ValueError(
                f"Invalid number of operands for {cls.__name__}: "
                f"expected {cls.n_operands}, got {len(children_nettypes)}"
            )
        if "node" in children_nettypes and "edge" in children_nettypes:
            return None  # node 和 edge 不能相互运算
        elif "node" in children_nettypes:
            return "node" # 有 node 时返回 node
        elif "edge" in children_nettypes:
            return "edge" # 有 edge 时返回 edge
        elif "scalar" in children_nettypes:
            return "scalar" # 只有 scalar 时返回 scalar
        else:
            return None # 没有有效的 nettype