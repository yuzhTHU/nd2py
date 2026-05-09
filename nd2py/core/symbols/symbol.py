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
        """Initialize a Symbol node.

        This constructor sets the nettype, sanitizes and attaches child
        operands, and then triggers a nettype inference pass on the whole
        expression tree.

        Args:
            *operands: Child operands of this symbol. The number of operands
                must match ``n_operands`` of the concrete subclass. Non-symbol
                scalar values are automatically wrapped as ``Number`` symbols.
            nettype (Optional[NetType | Set[NetType]]): Nettype constraint for
                this symbol, such as ``"node"``, ``"edge"``, or ``"scalar"``,
                or a set of allowed nettypes. If provided, it is propagated
                through the tree by ``infer_nettype()``.
        """
        NetTypeMixin.__init__(self, nettype=nettype)

        # 连接父代 (这里自底向上创建符号树, 没有 parent 需要连接)
        self.parent = None

        # 处理并连接子代 (子代在 _sanitize_operands 中被连接到父代)
        self.operands = self._sanitize_operands(operands)

        # 触发全树的 nettype 更新以传播约束 (如果有的话)
        # 大规模构造符号树时可能导致重复 inference 产生性能问题, 需要结合 no_nettype_inference 上下文管理器并在最后手动调用 infer_nettype() 以优化性能
        self.infer_nettype()

    def _sanitize_operands(self, operands: List["Symbol"]) -> List["Symbol"]:
        """Convert raw operands into a standardized list of ``Symbol`` objects.

        This method fills missing operands with ``Empty``, wraps numeric values
        as ``Number`` symbols, and ensures that each child has a unique parent
        by copying shared subexpressions when necessary.

        Args:
            operands (List[Symbol]): Raw operand list passed to the
                constructor.

        Returns:
            List[Symbol]: Sanitized list of operands attached to this symbol.

        Raises:
            ValueError: If the number of operands does not match
                ``self.n_operands``.
            TypeError: If an operand is neither a ``Symbol`` nor a valid
                numeric value.
        """
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
        """Return a deep copy of this symbol.

        The copied symbol has the same tree structure and values as the
        original but does not share ``parent`` links, so it can be safely
        inserted into a different expression tree.

        Returns:
            Symbol: A deep copy of the current symbol.
        """
        from ..basic import GetCopy

        return GetCopy()(self)

    def get_numbers(
        self,
        fitable_only: bool = False,
        float_only: bool = False,
        scalar_only: bool = False,
    ) -> List["Number"]:
        """Collect all ``Number`` nodes contained in this symbol.

        Traverses the expression tree in preorder and returns all numeric
        nodes that satisfy the given filters.

        Args:
            fitable_only (bool, optional): If True, return only numbers
                marked as fitable (trainable) parameters. Defaults to False.
            float_only (bool, optional): If True, exclude integer-like values
                (for example exponents that should remain fixed). Defaults to
                False.
            scalar_only (bool, optional): If True, only consider scalar
                numbers (nettype ``"scalar"``). Defaults to False.

        Returns:
            List[Number]: List of numeric symbol nodes that match the filters.
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
        """Return numeric parameter values contained in this symbol.

        This is a convenience wrapper over :meth:`get_numbers` that extracts
        the underlying scalar values from ``Number`` nodes.

        Args:
            fitable_only (bool, optional): If True, return only parameters
                associated with fitable numbers. Defaults to False.
            float_only (bool, optional): If True, exclude integer-like
                parameters. Defaults to False.

        Returns:
            List[float]: Flat list of parameter values in traversal order.
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
        """Assign new numeric parameter values to this symbol.

        The values in ``params`` are consumed in the same order as produced
        by :meth:`get_parameters` with the same filter options.

        Args:
            params (List[float]): New parameter values to assign.
            fitable_only (bool, optional): If True, only update fitable
                parameters and leave others unchanged. Defaults to False.
            float_only (bool, optional): If True, only update non-integer
                parameters. Defaults to False.

        Raises:
            ValueError: If the length of ``params`` does not match the number
                of parameters selected by the filters.
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
        """Default nettype mapping rule for symbol subclasses.

        The default behavior enforces that ``"node"`` and ``"edge"`` nettypes
        cannot be mixed. If only scalars are present, the result is
        ``"scalar"``; otherwise it follows the presence of ``"node"`` or
        ``"edge"``.

        Args:
            *children_nettypes (NetType): Nettypes of the child operands.

        Returns:
            Optional[NetType]: Inferred nettype for the parent symbol, or
            ``None`` if the combination is invalid or cannot be determined.

        Raises:
            ValueError: If the number of child nettypes does not match
                ``cls.n_operands``.
        """
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