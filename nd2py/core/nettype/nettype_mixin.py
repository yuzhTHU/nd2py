# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import itertools
from typing import Set
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Set, List, Optional, Literal
from ..context.nettype_inference import nettype_inference
from ...utils import classproperty

if TYPE_CHECKING: # 避免循环引用，仅用于类型检查
    from ..symbols import Symbol

# 定义核心类型
NetType = Literal["node", "edge", "scalar"]
ALL_NETTYPES: Set[NetType] = {"node", "edge", "scalar"}


class NetTypeMixin(ABC):
    """
    Mixin 类：负责维护 Symbol 的 nettype 候选集 (nettypes) 并处理约束传播。
        - 变量 (Variable), 数值 (Number) 和占位符 (Empty) 可以具有特定的 nettype, 表示该对象在网络中的角色 (节点、边或标量)
        - 不同运算符对操作符的 nettype 有不同的要求和限制, 所得输出的 nettype 也不同
        - 被指定为特定 nettype 的对象 (如 Variable(nettype="node")) 成为固定的锚点 (Anchor), 辅助其他对象通过约束传播确定自己的 nettype 候选集合

    宿主类 (Symbol) 必须提供以下属性和方法:
        - self.parent
        - self.operands
        - self.map_nettype(*children_nettypes: NetType) -> Optional[NetType]
    """

    # 通过 Type Hinting 提示宿主类必须有的属性, 不产生任何实际效果, 但方便 IDE 和类型检查器识别
    parent: Optional[Symbol]
    operands: List[Symbol]

    @classmethod
    @abstractmethod
    def map_nettype(cls, *children_nettypes: NetType) -> Optional[NetType]:
        """
        根据子节点 nettypes 推导自身 nettype, 返回 None 表示发现逻辑冲突.
        需要由具体的 Symbol 子类实现自己的映射逻辑
        """
        return None

    def __init__(self, nettype: Optional[NetType] = None):
        if isinstance(nettype, str) and nettype in ALL_NETTYPES: nettype = {nettype}
        # nettype 可以是 None 或 Set[NetType]
        # 如果用户提供了 nettype 作为硬约束, 将其用作限制其它节点的锚点
        self._assigned_nettypes: Set[NetType] = nettype.copy() if nettype else None
        # 此对象可取的 nettype 集合
        self._possible_nettypes: Set[NetType] = nettype.copy() if nettype else ALL_NETTYPES.copy()

    @property
    def possible_nettypes(self) -> Set[NetType]:
        """ 返回此对象可取的 nettype 集合 """
        return self._possible_nettypes
    
    @property
    def nettype(self) -> NetType|None:
        """ 如果此对象可取的 nettype 唯一，则返回该 nettype，否则返回 None 表示不确定。"""
        return next(iter(self._possible_nettypes)) if len(self._possible_nettypes) == 1 else None

    @possible_nettypes.setter
    def possible_nettypes(self, val: Set[NetType]):
        raise AttributeError("Cannot set possible_nettypes directly. Use nettype setter to set hard constraints or infer_nettype() to trigger inference.")
    
    @nettype.setter
    def nettype(self, val: NetType | Set[NetType] | None):
        val = {val} if isinstance(val, str) and val in ALL_NETTYPES else val
        # 性能优化：值未变则跳过
        if self._assigned_nettypes == val: return
        # 冲突检测
        if isinstance(val, set) and len(val) == 0: 
            raise ValueError(f"NetType Conflict: Possible nettypes became empty for {self}.")
        # 赋值并触发全树的 nettype 更新
        self._assigned_nettypes = val
        self.infer_nettype()

    def infer_nettype(self):
        """ 根据 _assigned_nettypes 推断整个符号树的 _possible_nettypes """
        from .inter_nettype import InferNettype
        if nettype_inference():
            infer_nettype = InferNettype()
            infer_nettype(self)

    @classmethod
    def get_nettype_range(cls) -> Set[NetType]:
        """ 获取此节点可能产生的所有 nettype 值域，并在首次调用时缓存到类属性中。 """
        if "_nettype_range" not in cls.__dict__: # cls.__dict__ 限定了只检查当前类自身的名称空间, 避免继承父类的 _nettype_range 标记
            possible_range = set()
            for combo in itertools.product(ALL_NETTYPES, repeat=cls.n_operands):
                if (res := cls.map_nettype(*combo)) is not None:
                    possible_range.add(res)
            cls._nettype_range = possible_range
        return cls._nettype_range
    
    @classproperty
    def nettype_range(cls) -> Set[NetType]:
        """ 获取此节点可能产生的所有 nettype 值域，并在首次调用时缓存到类属性中。 """
        return cls.get_nettype_range()
