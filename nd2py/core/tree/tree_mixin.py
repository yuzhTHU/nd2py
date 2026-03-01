# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Set, Optional, List, Tuple
from ..context.warn_once import warn_once
from ..nettype.nettype_mixin import NetType

if TYPE_CHECKING:
    # 避免循环引用，仅用于类型检查
    from ..symbols import Symbol, Variable


class TreeMixin:
    """
    Mixin 类：负责维护 Symbol 作为一棵符号树的结构关系 (parent, operands) 并提供树形操作方法。
    宿主类必须提供以下属性:
        - self.n_operands
        - self.parent
        - self.operands
    """
    n_operands: int
    parent: Optional["Symbol"]
    operands: List["Symbol"]
    _candidates: Set[NetType]

    def iter_preorder(self):
        """Non-recursive preorder traversal of the Symbol tree using an explicit stack."""
        from .iter_preorder import IterPreorder
        return IterPreorder()(self)

    def iter_postorder(self):
        """Postorder traversal of the Symbol tree."""
        from .iter_postorder import IterPostorder
        return IterPostorder()(self)

    def replace(self, child: "Symbol", other: "Symbol"):
        """Replace current expression (or subexpression denoted by child) with another expression."""
        if not any(child is op for op in self.iter_preorder()):
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
        self.infer_nettype()  # 可能需要更新 nettype
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
    