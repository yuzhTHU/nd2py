from typing import Set, Optional
from .symbols import Symbol
from ..nettype import NetType


class Variable(Symbol):
    n_operands = 0

    def __init__(self, name, nettype: NetType = "scalar"):
        super().__init__(nettype=nettype)
        self.name = name

    def map_nettype(self) -> Optional[NetType]:
        return self.nettype
