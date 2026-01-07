from typing import Set
from .symbols import Symbol
from ..nettype import NetType


class Variable(Symbol):
    n_operands = 0

    def __init__(self, name, nettype: NetType = "scalar"):
        super().__init__(nettype=nettype)
        self.name = name

    def nettype_range(self) -> Set[NetType]:
        # Since it has no operands, it cannot give a nettype different
        # from self.nettype by adjusting operands nettype combinations.
        return {self.nettype}
