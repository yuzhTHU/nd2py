# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from typing import Optional
from .symbols import Symbol
from ..nettype import NetType

__all__ = ["Empty"]


class Empty(Symbol):
    n_operands = 0

    def __init__(self, nettype: Optional[NetType] = None):
        super().__init__(nettype=nettype)
        self.operands = []
        self.parent = None

    def map_nettype(self) -> Optional[NetType]:
        # 这个方法不应该被调用
        raise RuntimeError("Empty 的 map_nettype 不应该被调用，因为它不参与 nettype 推导。")