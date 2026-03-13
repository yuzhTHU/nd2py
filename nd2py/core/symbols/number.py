# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import numpy as np
from typing import Set, Optional
from .symbol import Symbol, is_number
from ..nettype import NetType
from ..context.set_fitable import set_fitable

__all__ = ["Number"]


class Number(Symbol):
    n_operands = 0

    def __init__(self, value, nettype: NetType = "scalar", fitable=None):
        super().__init__(nettype=nettype)
        if isinstance(value, Number):
            if fitable is None: fitable = value.fitable
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

    def map_nettype(self) -> Optional[NetType]:
        return self.nettype

    def get_nettype_range(self) -> Set[NetType]:
        """ 获取此节点可能产生的所有 nettype 值域，并在首次调用时缓存到类属性中。 """
        return {self.nettype}
    
    @property
    def nettype_range(self) -> Set[NetType]:
        """ 获取此节点可能产生的所有 nettype 值域，并在首次调用时缓存到类属性中。 """
        return self.get_nettype_range()
