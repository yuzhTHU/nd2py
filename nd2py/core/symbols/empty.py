from typing import Optional
from .symbols import Symbol
from ..nettype import NetType


class Empty(Symbol):
    n_operands = 0

    def __init__(self, nettype: Optional[NetType] = None):
        self.operands = []
        self.parent = None
        self.nettype = nettype
