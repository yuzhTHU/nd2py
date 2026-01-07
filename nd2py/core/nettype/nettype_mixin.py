from __future__ import annotations
from typing import TYPE_CHECKING, Set, List, Optional, Literal

if TYPE_CHECKING:
    # 避免循环引用，仅用于类型检查
    from ..symbols import Symbol

# 定义核心类型
NetType = Literal["node", "edge", "scalar"]
ALL_NETTYPES: Set[NetType] = {"node", "edge", "scalar"}


class NetTypeMixin:
    """
    Mixin 类：负责维护 Symbol 的 nettype 候选集 (candidates) 并处理约束传播。
    宿主类 (Symbol) 必须提供: self.parent, self.operands
    以及重写 _update_self_from_children 和 get_allowed_nettypes_for_child 方法。
    """

    # 通过 Type Hinting 提示宿主类必须有的属性, 不产生任何实际效果, 但方便 IDE 和类型检查器识别
    parent: Optional[Symbol]
    operands: List[Symbol]

    def __init__(self, fixed_nettype: Optional[NetType] = None):
        # 如果指定了 fixed_nettype，则作为硬约束 (锚点)；否则默认为全集
        _initial_candidates = {fixed_nettype} if fixed_nettype else ALL_NETTYPES.copy()
        # 用户提供的硬约束 (锚点)
        self._initial_candidates: Set[NetType] = _initial_candidates
        # 此对象可取的 nettype 集合
        self._candidates: Set[NetType] = _initial_candidates.copy()

    @property
    def candidates(self) -> Set[NetType]:
        return self._candidates

    @candidates.setter
    def candidates(self, val: Set[NetType]):
        # 性能优化：值未变则跳过
        if self._candidates == val:
            return
        # 冲突检测
        if not val:
            raise ValueError(f"NetType Conflict: Candidates became empty for {self}.")
        # 赋值并触发强制传播 (修改 _candidates 后 propagate 内部的 diff 检查会失效, 必须强制推送)
        self._candidates = val
        self.propagate(force_push=True)
    
    @property
    def nettype(self) -> Optional[NetType]:
        """如果 candidates 唯一则返回该 nettype，否则返回 None 表示不确定。"""
        if len(self._candidates) == 1:
            return next(iter(self._candidates))
        else:
            return None

    def propagate(self, force_push: bool = False) -> bool:
        """
        核心传播逻辑。
        :param force_push: 是否跳过变更检测，强制通知 Parent 和 Children。
                           (通常由 setter 或 tree editor 触发)
        :return: bool, 如果发生不可调和的冲突返回 False
        """
        old_candidates = self._candidates.copy()

        # Step 1: 自检 (Self-Check): 根据子节点的状态收缩自己的 candidates
        if not self._update_self_from_children():
            return False
        # Step 2: 扩散 (Push Constraints)
        if force_push or (self._candidates != old_candidates):
            # 2.1 向上通知父节点更新 (Top-up)
            if self.parent:
                self.parent.propagate()
            # 2.2 向下限制其它子节点 (Top-down Context)
            for child in self.operands:
                self._push_constraint_to_child(child)
        return True

    def set_parent_with_constraint(self, parent: Symbol):
        """
        [逻辑连接] 子节点认父，拉取父代约束。
        通常在 Symbol.__init__ 或 set_operand 中被调用。
        """
        self.parent = parent
        # 获取父代允许的类型并应用约束 (这会自动触发 setter -> propagate -> notify neighbors)
        self.candidates &= parent.get_allowed_nettypes_for_child(self)

    def _push_constraint_to_child(self, child: Symbol):
        """将当前的 Context 约束强加给子节点"""
        child.candidates &= self.get_allowed_nettypes_for_child(child)

    # =========================================================
    # 需要由 Symbol 子类重写的业务钩子 (Business Hooks)
    # =========================================================

    def _update_self_from_children(self) -> bool:
        """
        [Bottom-up Logic] 根据子节点推导自身。
        返回 False 表示发现逻辑冲突。
        """
        return True

    def get_allowed_nettypes_for_child(self, child: Symbol) -> Set[NetType]:
        """[Top-down Logic] 根据自身和兄弟节点，限制特定 child。"""
        return ALL_NETTYPES.copy()
