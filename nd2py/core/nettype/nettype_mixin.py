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
    """Mixin that manages nettype candidates and constraint propagation.

    Symbols can carry a network type (nettype) describing their role in a
    computational graph, such as ``"node"``, ``"edge"`` or ``"scalar"``.
    This mixin tracks:

    - User-assigned nettype anchors (hard constraints).
    - The set of possible nettypes each symbol can take.
    - Propagation of constraints across the expression tree.

    The host class (typically ``Symbol``) is expected to provide:

    - ``self.parent``: Reference to the parent symbol in the tree.
    - ``self.operands``: List of child symbols.
    - ``self.map_nettype(*children_nettypes: NetType) -> Optional[NetType]``:
      Class method that maps child nettypes to a result nettype for the
      operator.
    """

    # 通过 Type Hinting 提示宿主类必须有的属性, 不产生任何实际效果, 但方便 IDE 和类型检查器识别
    parent: Optional[Symbol]
    operands: List[Symbol]

    @classmethod
    @abstractmethod
    def map_nettype(cls, *children_nettypes: NetType) -> Optional[NetType]:
        """Map child nettypes to the operator's resulting nettype.

        This method defines the nettype semantics of a symbol class. Concrete
        subclasses must implement their own mapping logic.

        Args:
            *children_nettypes (NetType): Nettypes of each operand, in order.

        Returns:
            Optional[NetType]: Inferred nettype of the operator, or ``None``
            if the combination is invalid or cannot be resolved.
        """
        return None

    def __init__(self, nettype: Optional[NetType] = None):
        """Initialize nettype state for a symbol-like object.

        The initial nettype can be provided as a single value or a set of
        allowed values. When specified, it acts as a hard constraint (anchor)
        that guides later nettype inference.

        Args:
            nettype (Optional[NetType | Set[NetType]]): Initial nettype
                constraint. If a string in ``ALL_NETTYPES``, it is converted
                to a singleton set; if ``None``, all nettypes are initially
                allowed.
        """
        if isinstance(nettype, str) and nettype in ALL_NETTYPES: nettype = {nettype}
        # nettype 可以是 None 或 Set[NetType]
        # 如果用户提供了 nettype 作为硬约束, 将其用作限制其它节点的锚点
        self._assigned_nettypes: Set[NetType] = nettype.copy() if nettype else None
        # 此对象可取的 nettype 集合
        self._possible_nettypes: Set[NetType] = nettype.copy() if nettype else ALL_NETTYPES.copy()

    @property
    def possible_nettypes(self) -> Set[NetType]:
        """Return the set of possible nettypes for this object.

        Returns:
            Set[NetType]: Current candidate nettypes that are consistent with
            all known constraints.
        """
        return self._possible_nettypes
    
    @property
    def nettype(self) -> NetType|None:
        """Return the resolved nettype if unique.

        When the candidate set contains exactly one element, this property
        returns that nettype. Otherwise it returns ``None`` to indicate that
        the nettype is still ambiguous.

        Returns:
            Optional[NetType]: Unique nettype if determined, otherwise
            ``None``.
        """
        return next(iter(self._possible_nettypes)) if len(self._possible_nettypes) == 1 else None

    @possible_nettypes.setter
    def possible_nettypes(self, val: Set[NetType]):
        raise AttributeError("Cannot set possible_nettypes directly. Use nettype setter to set hard constraints or infer_nettype() to trigger inference.")
    
    @nettype.setter
    def nettype(self, val: NetType | Set[NetType] | None):
        """Set a hard nettype constraint and trigger inference.

        The value is normalized to a set of allowed nettypes and stored as
        an assigned constraint. An empty set is treated as a conflict and
        results in an error. On successful assignment, a full-tree nettype
        inference pass is started.

        Args:
            val (NetType | Set[NetType] | None): New nettype constraint, or
                ``None`` to clear the hard constraint.

        Raises:
            ValueError: If ``val`` is an empty set, which indicates a
                nettype conflict.
        """
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
        """Infer possible nettypes for the entire expression tree.

        This method uses the currently assigned nettype constraints as
        anchors and propagates them through the symbol tree to update
        ``_possible_nettypes`` of all involved symbols.
        """
        from .inter_nettype import InferNettype
        if nettype_inference():
            infer_nettype = InferNettype()
            infer_nettype(self)

    @classmethod
    def get_nettype_range(cls) -> Set[NetType]:
        """Compute and cache the full nettype range for this operator class.

        The nettype range is the set of all possible result nettypes that can
        be produced by this operator, over all combinations of valid child
        nettypes. The result is cached on the class to avoid recomputation.

        Returns:
            Set[NetType]: Set of all nettypes that this operator can yield.
        """
        if "_nettype_range" not in cls.__dict__: # cls.__dict__ 限定了只检查当前类自身的名称空间, 避免继承父类的 _nettype_range 标记
            possible_range = set()
            for combo in itertools.product(ALL_NETTYPES, repeat=cls.n_operands):
                if (res := cls.map_nettype(*combo)) is not None:
                    possible_range.add(res)
            cls._nettype_range = possible_range
        return cls._nettype_range
    
    @classproperty
    def nettype_range(cls) -> Set[NetType]:
        """Class-level cached nettype range for this operator.

        Returns:
            Set[NetType]: Set of all nettypes that this operator can yield.
        """
        return cls.get_nettype_range()
