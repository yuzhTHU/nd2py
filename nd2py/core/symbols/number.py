import numpy as np
from typing import Set
from .symbols import Symbol, is_number
from ..nettype import NetType
from ..context.set_fitable import set_fitable


class Number(Symbol):
    n_operands = 0

    def __init__(self, value, nettype: NetType = "scalar", fitable=None):
        super().__init__(nettype=nettype)
        if isinstance(value, Number):
            fitable = fitable or value.fitable
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
