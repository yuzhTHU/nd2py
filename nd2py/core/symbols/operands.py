import warnings
from functools import reduce
from typing import Optional, List, Set
from .empty import Empty
from .symbols import Symbol
from ..nettype import NetType
from ..context.warn_once import warn_once


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
        if operand_types[0] == "scalar" and warn_once("readout_scalar"):
            if warn_once("inconsistent_nettype"):
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
