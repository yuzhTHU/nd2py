from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Set, Optional, List, Tuple
from ..context import warn_once
from ..nettype.nettype_mixin import NetType

if TYPE_CHECKING:
    # 避免循环引用，仅用于类型检查
    from ..symbols import Symbol, Variable


class TreeMixin:
    parent: Optional["Symbol"]
    operands: List["Symbol"]
    _candidates: Set[NetType]

    def iter_preorder(self):
        """Non-recursive preorder traversal of the Symbol tree using an explicit stack."""
        from ..iteration.iter_preorder import IterPreorder

        return IterPreorder()(self)

    def iter_postorder(self):
        """Postorder traversal of the Symbol tree."""
        from ..iteration.iter_postorder import IterPostorder

        return IterPostorder()(self)

    def replace(self, child: "Symbol", other: "Symbol"):
        """Replace current expression (or subexpression denoted by child) with another expression."""
        if not any(child == op for op in self.iter_preorder()):
            raise ValueError(
                f"Cannot replace '{child}' because it is not a subexpression of '{self}'"
            )
        if self.parent is not None:
            raise ValueError(
                f"Cannot replace subexpression of '{self}' because it is a subexpression of another Symbol"
            )
        # Ensure that 'other' is not a subexpression of another Symbol
        if other.parent is not None:
            other = other.copy()
        if self == child:
            # Replace the whold expression of self with other
            # This operation is allowed but may cause problems, especially when self is an item of a list
            # in which user need to update the list with the return value manually.
            if warn_once("replace_root_expression"):
                warnings.warn(
                    "You are replacing the root expression itself. Make sure to reassign the result, "
                    "otherwise external references (e.g. in lists or variables) still point to the old object.",
                    category=UserWarning,
                    stacklevel=2,
                )
            return other
        child_idx = child.parent.operands.index(child)
        child.parent.operands[child_idx] = other
        other.parent, child.parent = child.parent, None
        return self

    def path_to(self, child: "Symbol") -> str:
        """Get the path from self to child."""
        def iter(f, path=tuple([])):
            if f is child: return path
            for idx, op in enumerate(f.operands):
                result = iter(op, path + (idx,))
                if result is not None:
                    return result
        path = iter(self)
        return path
            
    def get_path(self, path: Tuple[int]) -> 'Symbol':
        """Get the subexpression at the specified path."""
        current = self
        for idx in path:
            if not (0 <= idx < len(current.operands)):
                raise IndexError(
                    f"Index {idx} out of range for operands of {current} with {len(current.operands)} operands."
                )
            current = current.operands[idx]
        return current
    
    def set_operand(self, index: int, child: "Symbol"):
        if child.parent:
            if not isinstance(child, Variable) and warn_once("subexpression_with_parent"):
                warnings.warn(
                    f"The object '{child}' cannot serve as a subexpression in multiple locations. "
                    f"It will be copied to avoid this behavior.",
                    category=UserWarning,
                    stacklevel=2,
                )
            child = child.copy()
        self.operands[index] = child
        child.set_parent_with_constraint(self)

    def replace_operand(self: Symbol, index: int, new_child: Symbol):
        """
        [原子操作] 替换子节点。
        自动处理：断开连接 -> 递归松弛 -> 建立新连接 -> 注入约束。
        """
        # 1. 临时占位 (断开物理连接)
        # 防止在松弛计算时，self 依然读取到旧 Child 的约束
        from ..symbols import Empty
        self.operands[index] = Empty() 
        
        # 2. 祖先链松弛 (Relaxation)
        # 这一步清除了旧节点留下的"幽灵约束"
        self._relax_ancestry(self)
        
        # 3. 正式连接 (Connect & Constrain)
        # 此时 self 已经是干净的状态，会根据新 Child 重新计算约束
        self.set_operand(index, new_child)

    def _relax_ancestry(self: Symbol):
        """ 从 self 开始向上找到根节点，并自顶向下重置约束路径 """
        # 1. 自 self 向上收集路径
        path = []
        curr = self
        while curr:
            path.append(curr)
            curr = curr.parent
        
        # 2. 自顶向下重置
        for ancestor in reversed(path):
            # [关键技巧] 绕过 setter 直接操作 _candidates, 防止在子节点重置之前触发 propagate，导致再次被旧子节点锁死
            ancestor._candidates = ancestor._initial_candidates.copy()
            
            # A. 重新吸收父节点约束
            if ancestor.parent:
                ancestor._candidates &= ancestor.parent.get_allowed_nettypes_for_child(ancestor)
            
            # [关键技巧] B. 先把所有孩子重置干净！
            # 必须在 propagate 之前做这一步
            for child in ancestor.operands:
                TreeMixin._recursive_reset_to_initial(child)
            
            # C. 现在大家都是干净的了，手动触发一次传播
            # 这会让 ancestor 重新评估孩子 (现在孩子是宽泛的)，并把宽泛的约束推给孩子
            ancestor.propagate(force_push=True)

    def _recursive_reset_to_initial(self: Symbol):
        """
        暴力递归重置子树状态到 Initial (单纯洗白数据)
        """
        self._candidates = self._initial_candidates.copy()
        for child in self.operands:
            TreeMixin._recursive_reset_to_initial(child)