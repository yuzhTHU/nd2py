# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from typing import Set, Optional
from .symbols import Symbol
from ..nettype import NetType, ALL_NETTYPES

__all__ = ["Variable"]


class Variable(Symbol):
    n_operands = 0

    def __init__(self, name, nettype: NetType = "scalar"):
        super().__init__(nettype=nettype)
        self.name = name

    def map_nettype(self) -> Optional[NetType]:
        return self.nettype

    def get_nettype_range(self) -> Set[NetType]:
        """ 获取此节点可能产生的所有 nettype 值域，并在首次调用时缓存到类属性中。 """
        return {self.nettype}
    
    @property
    def nettype_range(self) -> Set[NetType]:
        """ 获取此节点可能产生的所有 nettype 值域，并在首次调用时缓存到类属性中。 """
        return self.get_nettype_range()
