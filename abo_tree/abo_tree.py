from __future__ import annotations
from typing import List, Literal, Set, Optional, Type
import random

# 定义类型
Label = Literal['A', 'B', 'O']
ALL_LABELS: Set[Label] = {'A', 'B', 'O'}

class Node:
    def __init__(self):
        self.parent: Optional[Node] = None
        self.children: List[Node] = []
        # 这是“真实”的数据，永远维护可能性的集合
        self._candidates: Set[Label] = ALL_LABELS.copy()

    # === 需求二：显式化确定态 ===
    
    @property
    def candidates(self) -> Set[Label]:
        return self._candidates

    @candidates.setter
    def candidates(self, val: Set[Label]):
        self._candidates = val

    @property
    def is_determined(self) -> bool:
        return len(self._candidates) == 1

    @property
    def label(self) -> Optional[Label]:
        """如果是确定态，返回 label；否则返回 None"""
        if self.is_determined:
            return list(self._candidates)[0]
        return None

    def __repr__(self):
        # 使得打印结果能够区分 "确定态" 和 "叠加态"
        if self.is_determined:
            content = self.label
        else:
            # 按字母序排序，如 "{A,O}"
            content = "{" + ",".join(sorted(list(self._candidates))) + "}"
            
        if not self.children:
            return f"{self.__class__.__name__}({content})"
        
        children_str = ", ".join([repr(c) for c in self.children])
        return f"{self.__class__.__name__}({content}, children=[{children_str}])"

    # === 核心逻辑 ===

    def set_parent(self, parent: Node):
        self.parent = parent
        # 连接时，立即根据父节点当前的状态来收缩自己
        # 注意：这里我们取交集。Self 原本可能是 {A,B,O}，父节点可能限制只能是 {A,B}
        allowed_by_parent = parent.get_allowed_labels_for_child(self)
        self.candidates = self.candidates.intersection(allowed_by_parent)
        # 随后触发常规传播
        self.propagate()

    def propagate(self) -> bool:
        """
        约束传播：双向更新 (Top-down & Bottom-up)
        """
        old_candidates = self.candidates.copy()
        
        # 1. Bottom-up: 根据子节点更新自己
        if not self._update_self_from_children():
            return False

        # 2. Top-down & Sideways: 
        # 如果自己的 candidates 变少了，需要通知 Parent 和 Children
        if self.candidates != old_candidates:
            # 通知 Parent (向上回溯)
            if self.parent:
                if not self.parent.propagate(): return False
            
            # 通知 Children (向下约束)
            # 当 Parent 变了，需要告诉 Children "我现在变窄了，你们也要变窄"
            for child in self.children:
                allowed = self.get_allowed_labels_for_child(child)
                new_child_cand = child.candidates.intersection(allowed)
                if new_child_cand != child.candidates:
                    child.candidates = new_child_cand
                    if not child.candidates: return False
                    if not child.propagate(): return False
                    
        return True

    def _update_self_from_children(self) -> bool:
        """由子类实现：Child labels -> Self labels"""
        return True

    def get_allowed_labels_for_child(self, child: Node) -> Set[Label]:
        """
        === 需求一的核心 ===
        由子类实现：Context (Self + Siblings) -> Target Child labels
        给定 '我自己' 当前的允许范围，以及 '其他兄弟' 当前的允许范围，
        这个 'child' 允许取什么值？
        """
        return ALL_LABELS.copy()

# === 具体子类实现 ===

class Empty(Node):
    def __init__(self, candidates: Set[Label] = None):
        super().__init__()
        if candidates:
            self.candidates = candidates

    def _update_self_from_children(self):
        return len(self.candidates) > 0
    
    def __repr__(self):
        # Empty 特殊显示，方便调试
        if self.is_determined:
            return f"Empty({self.label})"
        cand_str = ",".join(sorted(list(self.candidates)))
        return f"Empty({{{cand_str}}})"

class Leaf(Node):
    def __init__(self, label: Label):
        super().__init__()
        self.candidates = {label}

class SingleChildNode(Node):
    """A2B, B2A, AB2O 的基类"""
    def __init__(self, child: Node):
        super().__init__()
        self.children = [child]
        child.parent = self
        self._init_self_constraint() # 初始化自身的强制约束 (如 A2B 必须是 B)

    def _init_self_constraint(self):
        pass

    def _update_self_from_children(self):
        # 逻辑：遍历 child 的可能性，看能映射出什么 self
        possible = set()
        for c_lbl in self.children[0].candidates:
            res = self._map_child_to_self(c_lbl)
            if res in self.candidates:
                possible.add(res)
        self.candidates = possible
        return len(self.candidates) > 0

    def get_allowed_labels_for_child(self, child: Node) -> Set[Label]:
        # 反向逻辑：遍历所有可能的 child label，看映射结果是否在 self.candidates 里
        allowed = set()
        for c_lbl in ALL_LABELS:
            res = self._map_child_to_self(c_lbl)
            if res in self.candidates:
                allowed.add(c_lbl)
        return allowed

    def _map_child_to_self(self, c_lbl: Label) -> Label:
        raise NotImplementedError

class A2B(SingleChildNode):
    def _init_self_constraint(self): self.candidates = {'B'}
    def _map_child_to_self(self, c): return 'B' if c in ['A', 'O'] else None

class B2A(SingleChildNode):
    def _init_self_constraint(self): self.candidates = {'A'}
    def _map_child_to_self(self, c): return 'A' if c in ['B', 'O'] else None

class AB2O(SingleChildNode):
    def _init_self_constraint(self): self.candidates = {'O'}
    def _map_child_to_self(self, c): return 'O' if c in ['A', 'B'] else None

class Branch(Node):
    def __init__(self, left: Node, right: Node):
        super().__init__()
        self.children = [left, right]
        left.parent = self
        right.parent = self
        # Branch 自身初始没有任何硬性约束，取决于 children
    
    def _update_self_from_children(self):
        c1s = self.children[0].candidates
        c2s = self.children[1].candidates
        possible = set()
        
        for l1 in c1s:
            for l2 in c2s:
                # 互斥约束
                if {l1, l2} == {'A', 'B'}: continue
                
                # 映射逻辑
                res = 'O'
                if 'A' in {l1, l2}: res = 'A'
                if 'B' in {l1, l2}: res = 'B'
                
                if res in self.candidates:
                    possible.add(res)
        
        self.candidates = possible
        return len(self.candidates) > 0

    def get_allowed_labels_for_child(self, child: Node) -> Set[Label]:
        """
        Branch 的核心反向推导。
        Target Child 允许的值 = f(Parent 当前允许值, Sibling 当前允许值)
        """
        is_left = (child == self.children[0])
        sibling = self.children[1] if is_left else self.children[0]
        sibling_opts = sibling.candidates
        
        allowed_for_target = set()
        
        # 穷举 Target Child 可能取的所有值
        for target_val in ALL_LABELS:
            valid_combination_found = False
            
            # 必须存在至少一个 Sibling 的取值，使得 (Target, Sibling) -> Parent
            for sib_val in sibling_opts:
                # 1. 检查互斥
                if {target_val, sib_val} == {'A', 'B'}:
                    continue
                
                # 2. 计算合成结果
                res = 'O'
                if 'A' in {target_val, sib_val}: res = 'A'
                if 'B' in {target_val, sib_val}: res = 'B'
                
                # 3. 检查结果是否被 Parent 允许
                if res in self.candidates:
                    valid_combination_found = True
                    break # 只要有一种 sibling 配合成功，这个 target_val 就是合法的
            
            if valid_combination_found:
                allowed_for_target.add(target_val)
                
        return allowed_for_target