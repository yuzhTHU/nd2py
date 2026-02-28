import warnings
from functools import reduce
from typing import Optional, List, Set
from .empty import Empty
from .symbols import Symbol
from ..nettype import NetType
from ..context.warn_once import warn_once


class Add(Symbol):
    n_operands = 2


class Sub(Symbol):
    n_operands = 2


class Mul(Symbol):
    n_operands = 2


class Div(Symbol):
    n_operands = 2


class Pow(Symbol):
    n_operands = 2


class Max(Symbol):
    n_operands = 2


class Min(Symbol):
    n_operands = 2


class Identity(Symbol):
    n_operands = 1


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
    def map_nettype(cls, *children_nettypes: NetType) -> Optional[NetType]:
        if len(children_nettypes) != cls.n_operands:
            raise ValueError(
                f"Invalid number of operands for {cls.__name__}: "
                f"expected {cls.n_operands}, got {len(children_nettypes)}"
            )
        return "edge" if children_nettypes[0] != "edge" else None


class Targ(Sour):
    pass


class Aggr(Symbol):
    n_operands = 1

    @classmethod
    def map_nettype(cls, *children_nettypes: NetType) -> Optional[NetType]:
        if len(children_nettypes) != cls.n_operands:
            raise ValueError(
                f"Invalid number of operands for {cls.__name__}: "
                f"expected {cls.n_operands}, got {len(children_nettypes)}"
            )
        return "node" if children_nettypes[0] != "node" else None

class Rgga(Aggr):
    pass


class Readout(Symbol):
    n_operands = 1

    @classmethod
    def map_nettype(cls, *children_nettypes: NetType) -> Optional[NetType]:
        if len(children_nettypes) != cls.n_operands:
            raise ValueError(
                f"Invalid number of operands for {cls.__name__}: "
                f"expected {cls.n_operands}, got {len(children_nettypes)}"
            )
        return "scalar"
