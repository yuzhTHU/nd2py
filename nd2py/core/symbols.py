import warnings
import numpy as np
from copy import deepcopy
from functools import reduce
from typing import List, Literal, Optional, Tuple, Set
from .context.check_nettype import check_nettype
from .context.set_fitable import set_fitable

__all__ = [
    "NetType",
    "Symbol",
    "Empty",
    "Number",
    "Variable",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Max",
    "Min",
    "Sin",
    "Cos",
    "Tan",
    "Sec",
    "Csc",
    "Cot",
    "Log",
    "LogAbs",
    "Exp",
    "Abs",
    "Neg",
    "Inv",
    "Sqrt",
    "SqrtAbs",
    "Pow2",
    "Pow3",
    "Arcsin",
    "Arccos",
    "Arctan",
    "Sinh",
    "Cosh",
    "Tanh",
    "Coth",
    "Sech",
    "Csch",
    "Sigmoid",
    "Regular",
    "Sour",
    "Targ",
    "Aggr",
    "Rgga",
    "Readout",
]

NetType = Literal["node", "edge", "scalar"]


def _warn_once(warn_name, maxsize=1):
    """This function is used to limit the number of times a warning is issued"""
    if not hasattr(_warn_once, warn_name):
        setattr(_warn_once, warn_name, 0)
    else:
        setattr(_warn_once, warn_name, getattr(_warn_once, warn_name) + 1)
    return getattr(_warn_once, warn_name) < maxsize


def is_number(value, scalar_only=False):
    """Check if the value is a number or a numpy array"""
    import numbers
    import numpy as np

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


class Symbol(metaclass=SymbolMeta):
    # Number of operands for this Symbol, should be set in subclasses
    n_operands = None

    def __init__(self, *operands, nettype: NetType = None):
        self.parent = None

        operands = list(operands)
        # Fill operands with Empty() if not enough operands are provided
        if len(operands) == 0:
            operands = [Empty(nettype=None) for _ in range(self.n_operands)]
        # Convert numbers to Number instances
        for idx, op in enumerate(operands):
            if is_number(op, scalar_only=True):
                operands[idx] = Number(op, nettype="scalar")
        # Check that all operands are instances of Symbol
        for op in operands:
            assert isinstance(op, Symbol), f"Invalid operand type: {type(op)}"
        for idx, op in enumerate(operands):
            # Ensure operands is not a subexpression of another Symbol
            if op.parent is not None:
                operands[idx] = op = op.copy()
                # Do not print warnings for Variable, since this scenerio is rather common.
                if not isinstance(op, Variable):
                    if _warn_once("subexpression_with_parent"):
                        warnings.warn(
                            f"The object '{op}' cannot serve as a subexpression in multiple locations. It will be copied to avoid this behavior.",
                            category=UserWarning,
                            stacklevel=2,
                        )
            # Set the parent of each operand to self
            op.parent = self
        self.operands = operands

        # Set nettype to the one specified or inferred from operands
        implied_nettype = (
            self.map_nettype([op.nettype for op in operands])
            if len(operands)
            else nettype
        )
        self.nettype = nettype or implied_nettype
        # Ensure nettype of operands meets the requirements of the current Symbol
        if check_nettype() and implied_nettype == "invalid":
            raise ValueError(
                f"Cannot determine nettype for {type(self).__name__} with operands {operands}. "
            )
        # Ensure nettype is specified or can be inferred
        if check_nettype() and self.nettype is None:
            raise ValueError(
                f"The nettype of {type(self).__name__} is not specified, nor can it be inferred from operands."
            )
        # Ensure nettype is valid
        if (
            check_nettype()
            and self.nettype is not None
            and self.nettype not in self.nettype_range()
        ):
            raise ValueError(
                f"Invalid nettype '{nettype}' for {type(self).__name__}, while {self.nettype_range()} is desired."
            )
        # Ensure consistency of nettype if it is both specified and can be inferred
        if (
            check_nettype()
            and self.nettype is not None
            and implied_nettype is not None
            and self.nettype != implied_nettype
        ):
            if _warn_once("inconsistent_nettype"):
                warnings.warn(
                    f'You are trying to create a {type(self).__name__} with nettype "{self.nettype}" '
                    f"but the operands imply a different nettype: {implied_nettype}. "
                    "Please make sure this behavior is what you expect.",
                    category=UserWarning,
                    stacklevel=2,
                )
        # Ensure nettypes of operands are valid
        for op in operands:
            if (
                check_nettype()
                and op.nettype is not None
                and op.nettype not in self.replaceable_nettype(op)
            ):
                raise ValueError(
                    f"Invalid nettype '{op.nettype}' for subexpression '{op}' of {type(self).__name__}, "
                    f"while {self.replaceable_nettype(op)} is desired."
                )

    def __repr__(self):
        return self.to_str()

    def __str__(self):
        return self.to_str()

    def __len__(self):
        from .basic.get_length import GetLength

        return GetLength()(self)

    def __add__(self, other):
        if is_number(other):
            other = Number(other)
        return Add(self, other)

    def __radd__(self, other):
        if is_number(other):
            other = Number(other)
        return Add(other, self)

    def __sub__(self, other):
        if is_number(other):
            other = Number(other)
        return Sub(self, other)

    def __rsub__(self, other):
        if is_number(other):
            other = Number(other)
        return Sub(other, self)

    def __mul__(self, other):
        if is_number(other):
            other = Number(other)
        return Mul(self, other)

    def __rmul__(self, other):
        if is_number(other):
            other = Number(other)
        return Mul(other, self)

    def __truediv__(self, other):
        if is_number(other):
            other = Number(other)
        return Div(self, other)

    def __rtruediv__(self, other):
        if is_number(other):
            other = Number(other)
        return Div(other, self)

    def __pow__(self, other):
        if is_number(other):
            other = Number(other)
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
        if is_number(other):
            other = Number(other)
        return Pow(other, self)

    def __neg__(self):
        return Neg(self)

    @classmethod
    def map_nettype(cls, operand_types: List[Optional[NetType]]) -> Optional[NetType]:
        """Determine the nettype of this Symbol based on its operands."""
        if len(operand_types) != cls.n_operands:
            raise ValueError(
                f"Invalid number of operands for {cls.__name__}: expected {cls.n_operands}, got {len(operand_types)}"
            )
        elif "node" in operand_types and "edge" in operand_types:
            return "invalid"
        elif "node" in operand_types:
            return "node"
        elif "edge" in operand_types:
            return "edge"
        elif "scalar" in operand_types:
            return "scalar"
        elif all(op is None for op in operand_types):
            return None
        else:
            return "invalid"

    @classmethod
    def nettype_range(cls) -> Set[NetType]:
        """Possible nettypes of this Symbol under all operand nettype combinations."""
        return {"node", "edge", "scalar"}

    def replaceable_nettype(self, child: "Symbol" = None, strict=False) -> Set[NetType]:
        """Nettypes that can be used to replace this subexpression (or its child if specified)."""
        # If child is specified, it must be a subexpression of self
        if child is not None and child not in self.operands:
            raise ValueError(
                f"Cannot get replaceable nettype for {child} because it is not a subexpression of {self}"
            )
        # If strict mode is enabled, the returned nettype must ensure that .map_nettype(returned nettype) == parent.nettype,
        # which will result in operands of node-level Symbol not being replaced with scalar types, reducing the flexibility of genetic programming.
        if strict:
            raise NotImplementedError(
                f"{type(self).__name__}.replaceable_nettype() is not implemented for strict mode"
            )
        if child is not None:
            return {self.nettype, "scalar"}
        elif self.parent is not None:
            return self.parent.replaceable_nettype(self, strict=strict)
        else:
            # Not self.nettype_range() since the genetic programming expects the whole expression to be of the specified type
            return {self.nettype, "scalar"}

    def iter_preorder(self):
        """Non-recursive preorder traversal of the Symbol tree using an explicit stack."""
        from .iteration.iter_preorder import IterPreorder

        return IterPreorder()(self)

    def iter_postorder(self):
        """Postorder traversal of the Symbol tree."""
        from .iteration.iter_postorder import IterPostorder

        return IterPostorder()(self)

    def replace(self, child: "Symbol", other: "Symbol"):
        """Replace current expression (or subexpression denoted by child) with another expression."""
        if not any(child == op for op in self.iter_preorder()):
            raise ValueError(
                f"Cannot replace '{child}' because it is not a subexpression of '{self}'"
            )
        if self.parent is not None:
            raise ValueError(
                f"Cannot replace subexpression of '{self}' because it is a subexpression of another Symbol"
            )
        # Ensure that 'other' is not a subexpression of another Symbol
        if other.parent is not None:
            other = other.copy()
        if self == child:
            # Replace the whold expression of self with other
            # This operation is allowed but may cause problems, especially when self is an item of a list
            # in which user need to update the list with the return value manually.
            if _warn_once("replace_root_expression"):
                warnings.warn(
                    "You are replacing the root expression itself. Make sure to reassign the result, "
                    "otherwise external references (e.g. in lists or variables) still point to the old object.",
                    category=UserWarning,
                    stacklevel=2,
                )
            return other
        child_idx = child.parent.operands.index(child)
        child.parent.operands[child_idx] = other
        other.parent, child.parent = child.parent, None
        return self

    def copy(self):
        """Create a deep copy of the Symbol. The result will not inherit self.parent"""
        from .basic.get_copy import GetCopy

        return GetCopy()(self)

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
        from .printer.string_printer import StringPrinter

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
        from .printer.tree_printer import TreePrinter

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
        from .calc.numpy_calc import NumpyCalc

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
        from .calc.torch_calc import TorchCalc

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
        from .transform.split_by_add import SplitByAdd

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
        from .transform.split_by_mul import SplitByMul

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
        from .transform.fix_nettype import FixNetType

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


class Empty(Symbol):
    n_operands = 0

    def __init__(self, nettype: Optional[NetType] = None):
        self.operands = []
        self.parent = None
        self.nettype = nettype


class Number(Symbol):
    n_operands = 0

    def __init__(self, value, nettype: NetType = "scalar", fitable=None):
        super().__init__(nettype=nettype)
        if isinstance(value, Number):
            value = value.value
        if fitable is None:
            fitable = set_fitable()
        self.value = value
        self.fitable = fitable

    def __eq__(self, other) -> bool:
        if is_number(other):
            return np.all(self.value == other)
        elif isinstance(other, Number):
            return np.all(self.value == other.value)

    def nettype_range(self) -> Set[NetType]:
        # Since it has no operands, it cannot give a nettype different
        # from self.nettype by adjusting operands nettype combinations.
        return {self.nettype}


class Variable(Symbol):
    n_operands = 0

    def __init__(self, name, nettype: NetType = "scalar"):
        super().__init__(nettype=nettype)
        self.name = name

    def nettype_range(self) -> Set[NetType]:
        # Since it has no operands, it cannot give a nettype different
        # from self.nettype by adjusting operands nettype combinations.
        return {self.nettype}


class Add(Symbol):
    n_operands = 2

    def __init__(self, *operands, nettype: NetType = None):
        if len(operands) < 2:
            operands = list(operands) + [
                Empty(nettype=None) for _ in range(2 - len(operands))
            ]
        other = reduce(lambda x, y: Add(x, y), operands[:-1])
        super().__init__(other, operands[-1], nettype=nettype)


class Sub(Symbol):
    n_operands = 2


class Mul(Symbol):
    n_operands = 2

    def __init__(self, *operands, nettype: NetType = None):
        if len(operands) < 2:
            operands = list(operands) + [
                Empty(nettype=None) for _ in range(2 - len(operands))
            ]
        other = reduce(lambda x, y: Mul(x, y), operands[:-1])
        super().__init__(other, operands[-1], nettype=nettype)


class Div(Symbol):
    n_operands = 2


class Pow(Symbol):
    n_operands = 2


class Max(Symbol):
    n_operands = 2

    def __init__(self, *operands, nettype: NetType = None):
        if len(operands) < 2:
            operands = list(operands) + [
                Empty(nettype=None) for _ in range(2 - len(operands))
            ]
        other = reduce(lambda x, y: Max(x, y), operands[:-1])
        super().__init__(other, operands[-1], nettype=nettype)


class Min(Symbol):
    n_operands = 2

    def __init__(self, *operands, nettype: NetType = None):
        if len(operands) < 2:
            operands = list(operands) + [
                Empty(nettype=None) for _ in range(2 - len(operands))
            ]
        other = reduce(lambda x, y: Min(x, y), operands[:-1])
        super().__init__(other, operands[-1], nettype=nettype)


class Sin(Symbol):
    n_operands = 1


class Cos(Symbol):
    n_operands = 1


class Tan(Symbol):
    n_operands = 1


class Sec(Symbol):
    n_operands = 1


class Csc(Symbol):
    n_operands = 1


class Cot(Symbol):
    n_operands = 1


class Log(Symbol):
    n_operands = 1


class LogAbs(Symbol):
    n_operands = 1


class Exp(Symbol):
    n_operands = 1


class Abs(Symbol):
    n_operands = 1


class Neg(Symbol):
    n_operands = 1


class Inv(Symbol):
    n_operands = 1


class Sqrt(Symbol):
    n_operands = 1


class SqrtAbs(Symbol):
    n_operands = 1


class Pow2(Symbol):
    n_operands = 1


class Pow3(Symbol):
    n_operands = 1


class Arcsin(Symbol):
    n_operands = 1


class Arccos(Symbol):
    n_operands = 1


class Arctan(Symbol):
    n_operands = 1


class Sinh(Symbol):
    n_operands = 1


class Cosh(Symbol):
    n_operands = 1


class Tanh(Symbol):
    n_operands = 1


class Sech(Symbol):
    n_operands = 1


class Csch(Symbol):
    n_operands = 1


class Coth(Symbol):
    n_operands = 1


class Sigmoid(Symbol):
    n_operands = 1


class Regular(Symbol):
    n_operands = 2


class Sour(Symbol):
    n_operands = 1

    @classmethod
    def map_nettype(
        cls, operand_types: Optional[List[NetType]] = None
    ) -> Optional[NetType]:
        """Determine the nettype of this Symbol based on its operands."""
        if len(operand_types) != cls.n_operands:
            raise ValueError(
                f"Invalid number of operands for {cls.__name__}: expected {cls.n_operands}, got {len(operand_types)}"
            )
        if operand_types[0] == "edge":
            return "invalid"
        return "edge"

    @classmethod
    def nettype_range(cls) -> Set[NetType]:
        """Possible nettypes of this Symbol under all operand nettype combinations."""
        return {"edge"}

    def replaceable_nettype(self, child: "Symbol" = None, strict=False) -> Set[NetType]:
        """Nettypes that can be used to replace this subexpression (or its child if specified)."""
        if child is not None and child not in self.operands:
            raise ValueError(
                f"Cannot get replaceable nettype for {child} because it is not a subexpression of {self}"
            )
        if strict:
            raise NotImplementedError(
                f"{type(self).__name__}.replaceable_nettype() is not implemented for strict mode"
            )
        if child is not None:
            # Changed here: because the operand of Sour and itself have different nettypes
            return {"node", "scalar"}
        elif self.parent is not None:
            return self.parent.replaceable_nettype(self, strict=strict)
        else:
            return {self.nettype, "scalar"}


class Targ(Sour):
    pass


class Aggr(Symbol):
    n_operands = 1

    @classmethod
    def map_nettype(
        cls, operand_types: Optional[List[NetType]] = None
    ) -> Optional[NetType]:
        """Determine the nettype of this Symbol based on its operands."""
        if len(operand_types) != cls.n_operands:
            raise ValueError(
                f"Invalid number of operands for {cls.__name__}: expected {cls.n_operands}, got {len(operand_types)}"
            )
        if operand_types[0] == "node":
            return "invalid"
        return "node"

    @classmethod
    def nettype_range(cls) -> Set[NetType]:
        """Possible nettypes of this Symbol under all operand nettype combinations."""
        return {"node"}

    def replaceable_nettype(self, child: "Symbol" = None, strict=False) -> Set[NetType]:
        """Nettypes that can be used to replace this subexpression (or its child if specified)."""
        if child is not None and child not in self.operands:
            raise ValueError(
                f"Cannot get replaceable nettype for {child} because it is not a subexpression of {self}"
            )
        if strict:
            raise NotImplementedError(
                f"{type(self).__name__}.replaceable_nettype() is not implemented for strict mode"
            )
        if child is not None:
            # Changed here: because the operand of Aggr and itself have different nettypes
            return {"edge", "scalar"}
        elif self.parent is not None:
            return self.parent.replaceable_nettype(self, strict=strict)
        else:
            return {self.nettype, "scalar"}


class Rgga(Aggr):
    pass


class Readout(Symbol):
    n_operands = 1

    @classmethod
    def map_nettype(
        cls, operand_types: Optional[List[NetType]] = None
    ) -> Optional[NetType]:
        """Determine the nettype of this Symbol based on its operands."""
        if len(operand_types) != cls.n_operands:
            raise ValueError(
                f"Invalid number of operands for {cls.__name__}: expected {cls.n_operands}, got {len(operand_types)}"
            )
        if operand_types[0] == "scalar" and _warn_once("readout_scalar"):
            if _warn_once("inconsistent_nettype"):
                warnings.warn(
                    f"Trying to apply {cls.__name__} to a 'scalar' variable, which have no effect. ",
                    category=UserWarning,
                    stacklevel=2,
                )
        return "scalar"

    @classmethod
    def nettype_range(cls) -> Set[NetType]:
        """Possible nettypes of this Symbol under all operand nettype combinations."""
        return {"scalar"}

    def replaceable_nettype(self, child: "Symbol" = None, strict=False) -> Set[NetType]:
        """Nettypes that can be used to replace this subexpression (or its child if specified)."""
        if child is not None and child not in self.operands:
            raise ValueError(
                f"Cannot get replaceable nettype for {child} because it is not a subexpression of {self}"
            )
        if strict:
            raise NotImplementedError(
                f"{type(self).__name__}.replaceable_nettype() is not implemented for strict mode"
            )
        if child is not None:
            # scalar is possible to replace its operand, but not recommended
            return {"node", "edge", "scalar"}
        elif self.parent is not None:
            return self.parent.replaceable_nettype(self, strict=strict)
        else:
            return {self.nettype, "scalar"}
