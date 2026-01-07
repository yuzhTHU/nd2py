import random
from typing import List, Literal, Set, Optional, Type, Callable

# ==========================================
# 1. 核心数据结构与逻辑 (Core Classes)
# ==========================================

Label = Literal['A', 'B', 'O']
ALL_LABELS: Set[Label] = {'A', 'B', 'O'}

class Node:
    def __init__(self):
        self.parent: Optional[Node] = None
        self.children: List[Node] = []
        self._candidates: Set[Label] = ALL_LABELS.copy()

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
        if self.is_determined:
            return list(self._candidates)[0]
        return None

    def set_parent(self, parent: 'Node'):
        self.parent = parent
        # 连接时，获取父节点允许的上下文约束 (Context Constraint)
        allowed = parent.get_allowed_labels_for_child(self)
        self.candidates = self.candidates.intersection(allowed)
        self.propagate()

    def propagate(self) -> bool:
        """核心约束传播：双向更新"""
        old_candidates = self.candidates.copy()
        
        # 1. Bottom-up: 根据子节点推导自身
        if not self._update_self_from_children():
            return False # 冲突

        # 2. Top-down / Propagation: 自身变化通知邻居
        if self.candidates != old_candidates:
            # 向上通知 Parent
            if self.parent:
                if not self.parent.propagate(): return False
            
            # 向下通知 Children
            for child in self.children:
                allowed = self.get_allowed_labels_for_child(child)
                new_child_cand = child.candidates.intersection(allowed)
                if new_child_cand != child.candidates:
                    child.candidates = new_child_cand
                    if not child.candidates: return False
                    if not child.propagate(): return False
                    
        return True

    def _update_self_from_children(self) -> bool:
        return True

    def get_allowed_labels_for_child(self, child: 'Node') -> Set[Label]:
        return ALL_LABELS.copy()

    def __repr__(self):
        if self.is_determined:
            content = self.label
        else:
            content = "{" + ",".join(sorted(list(self._candidates))) + "}"
        
        if not self.children:
            return f"{self.__class__.__name__}({content})"
        children_str = ", ".join([repr(c) for c in self.children])
        return f"{self.__class__.__name__}({content})[{children_str}]"

# --- 子类实现 ---

class Empty(Node):
    def _update_self_from_children(self):
        return len(self.candidates) > 0
    def __repr__(self):
        if self.is_determined: return f"Empty({self.label})"
        return f"Empty({{{','.join(sorted(list(self.candidates)))}}})"

class Leaf(Node):
    def __init__(self, label: Label):
        super().__init__()
        self.candidates = {label}
    def _update_self_from_children(self):
        return len(self.candidates) == 1

class Branch(Node):
    def __init__(self, c1: Node, c2: Node):
        super().__init__()
        self.children = [c1, c2]
        # 同样使用 set_parent 建立连接
        c1.set_parent(self)
        c2.set_parent(self)
    
    def _update_self_from_children(self):
        # 规则：
        # 1. Child 不能 A+B
        # 2. A+A=A, A+O=A; B+B=B, B+O=B; O+O=O
        c1s, c2s = self.children[0].candidates, self.children[1].candidates
        possible = set()
        for l1 in c1s:
            for l2 in c2s:
                if {l1, l2} == {'A', 'B'}: continue # 互斥
                
                res = 'O'
                if 'A' in {l1, l2}: res = 'A'
                elif 'B' in {l1, l2}: res = 'B'
                
                if res in self.candidates:
                    possible.add(res)
        self.candidates = possible
        return len(self.candidates) > 0

    def get_allowed_labels_for_child(self, child: Node) -> Set[Label]:
        # 反向推导：根据 Self 和 Sibling 推导 Child
        is_left = (child == self.children[0])
        sibling = self.children[1] if is_left else self.children[0]
        
        allowed = set()
        for target in ALL_LABELS:
            # 只要能在 sibling 中找到一个配对，使得结果在 self.candidates 中，则 target 合法
            for sib in sibling.candidates:
                if {target, sib} == {'A', 'B'}: continue # 互斥
                
                res = 'O'
                if 'A' in {target, sib}: res = 'A'
                elif 'B' in {target, sib}: res = 'B'
                
                if res in self.candidates:
                    allowed.add(target)
                    break 
        return allowed

class SingleChildNode(Node):
    def __init__(self, child: Node):
        super().__init__()
        # 1. 先初始化自身的硬约束 (如 B2A 必须是 A)
        self._init_constraint() 
        
        self.children = [child]
        # 2. 使用 set_parent 替代直接赋值
        # 这会立即触发: child.candidates = child.candidates ∩ self.get_allowed_labels_for_child(child)
        child.set_parent(self)

    def _init_constraint(self): pass
    def _map(self, c_lbl): return None
    
    def _update_self_from_children(self):
        possible = set()
        for c in self.children[0].candidates:
            res = self._map(c)
            if res and res in self.candidates:
                possible.add(res)
        self.candidates = possible
        return len(self.candidates) > 0
    
    def get_allowed_labels_for_child(self, child: Node) -> Set[Label]:
        allowed = set()
        for c in ALL_LABELS:
            res = self._map(c)
            if res and res in self.candidates:
                allowed.add(c)
        return allowed

class A2B(SingleChildNode):
    def _init_constraint(self): self.candidates = {'B'}
    def _map(self, c): return 'B' if c in ['A', 'O'] else None

class B2A(SingleChildNode):
    def _init_constraint(self): self.candidates = {'A'}
    def _map(self, c): return 'A' if c in ['B', 'O'] else None

class AB2O(SingleChildNode):
    def _init_constraint(self): self.candidates = {'O'}
    def _map(self, c): return 'O' if c in ['A', 'B'] else None


# ==========================================
# 2. 引擎实现 (Engines)
# ==========================================

class TopDownEngine:
    def __init__(self, root_label: Label):
        self.root = Empty()
        self.root.candidates = {root_label}
        print(f"[*] Initialized TopDownEngine with Root Target: {root_label}")

    def get_empties(self, node: Node, res: List[Empty]):
        if isinstance(node, Empty): res.append(node)
        for c in node.children: self.get_empties(c, res)

    def run(self):
        step = 1
        while True:
            empties = []
            self.get_empties(self.root, empties)
            if not empties:
                print("\n[Done] Generation Complete!")
                print(f"Final Tree: {self.root}")
                return self.root

            print(f"\n--- Step {step}: {len(empties)} empty slots remaining ---")
            print(f"Current Tree: {self.root}")
            
            # 策略：随机选一个 Empty
            target = random.choice(empties)
            print(f"Targeting: {target} (Parent: {target.parent.__class__.__name__ if target.parent else 'None'})")
            
            # 生成所有合法的 Move
            moves = self.get_valid_moves(target)
            if not moves:
                print("Error: No valid moves for this slot! (Dead end)")
                return None
            
            # 随机选一个 Move
            move_name, move_func = random.choice(moves)
            print(f"-> Action: Applying {move_name}")
            
            try:
                move_func()
            except Exception as e:
                print(f"Constraint Violation during move: {e}")
                return None
            
            step += 1

    def get_valid_moves(self, target: Empty):
        moves = []
        
        def apply(node_maker):
            new_node = node_maker()
            
            # [关键修正]
            # 因为 target 是 Empty，target.candidates 纯粹代表"环境/外部约束"。
            # 新节点必须遵守环境约束。
            # 这不会导致"自我约束"的死循环，因为 Empty 没有自我约束。
            intersection = new_node.candidates.intersection(target.candidates)
            
            if not intersection:
                # 比如环境要求 {A}, 但我们试图填入 A2B(强制B) -> 空集 -> 非法移动
                raise ValueError("Constraint Violation")
            
            new_node.candidates = intersection

            parent = target.parent
            if parent:
                idx = parent.children.index(target)
                parent.children[idx] = new_node
                # 再次连接 Parent，确保双向一致性
                new_node.set_parent(parent)
            else:
                self.root = new_node
                # Root 也要做一致性检查 (Propagate)
                if not new_node.propagate(): 
                    raise ValueError("Root conflict")
        # 1. Leaf
        for lbl in ['A', 'B', 'O']:
            if lbl in target.candidates:
                moves.append((f"Leaf({lbl})", lambda l=lbl: apply(lambda: Leaf(l))))
        
        # 2. Branch (Initializes with 2 Empties)
        # Branch 本身极其灵活，几乎总能填入，除非 Target 只能是某些特殊情况
        # 简单起见，只要 propagate 不报错就行
        moves.append(("Branch", lambda: apply(lambda: Branch(Empty(), Empty()))))
        
        # 3. Transformers
        # 只有当 target 允许 Transformer 的输出时才添加
        if 'B' in target.candidates:
            moves.append(("A2B", lambda: apply(lambda: A2B(Empty()))))
        if 'A' in target.candidates:
            moves.append(("B2A", lambda: apply(lambda: B2A(Empty()))))
        if 'O' in target.candidates:
            moves.append(("AB2O", lambda: apply(lambda: AB2O(Empty()))))
            
        return moves


class BottomUpEngine:
    @staticmethod
    def mask_node(node_to_replace: Node, keep_root_constraint=True) -> Empty:
        """
        将节点替换为 Empty。
        
        关键逻辑：
        1. 丢弃 node_to_replace 的"自我约束" (Self Constraint)。
        2. 重新从 Parent 获取"环境约束" (Context Constraint)。
        """
        parent = node_to_replace.parent
        new_empty = Empty() # 初始状态：{A, B, O} (无自我约束)
        
        if parent:
            # === 情况 A: 有父节点 ===
            # 直接挂载，利用 parent 计算环境约束
            idx = parent.children.index(node_to_replace)
            parent.children[idx] = new_empty
            
            # 这一步会自动计算 new_empty = {A,B,O} ∩ allowed_by_parent
            # 所以如果 parent 是 AB2O，new_empty 自动变成 {A, B}
            # 我们完全忽略了 node_to_replace 原本是 A 还是 B
            new_empty.set_parent(parent) 
            
        else:
            # === 情况 B: 根节点 ===
            # 这里的处理体现了你担心的点。
            # 如果 node_to_replace 是 Root=Leaf(A)，我们是否要保留 {A}?
            if keep_root_constraint:
                # 保留原全局目标：{A,B,O} ∩ {A} -> {A}
                new_empty.candidates = new_empty.candidates.intersection(node_to_replace.candidates)
            else:
                # 彻底重置：{A,B,O}
                pass
        
        return new_empty

    def run(self, target_label: Label, steps=10):
        print(f"\n[*] Initialized BottomUpEngine searching for: {target_label}")
        pool = [Leaf('A'), Leaf('B'), Leaf('O'), Leaf('A'), Leaf('O')] # 初始池
        
        for i in range(steps):
            print(f"--- Iteration {i+1} (Pool size: {len(pool)}) ---")
            
            # 检查是否成功
            matches = [n for n in pool if n.label == target_label and not isinstance(n, Leaf)]
            if matches and i > 2: # 稍微多生成几轮
                res = random.choice(matches)
                print(f"[Done] Found suitable candidate: {res}")
                return res
            
            # 随机组合
            op = random.choice(['Branch', 'A2B', 'B2A', 'AB2O'])
            try:
                if op == 'Branch':
                    if len(pool) < 2: continue
                    idx1, idx2 = random.sample(range(len(pool)), 2)
                    # Use pop to remove, or just sample to reuse? Reusing allows complex DAGs, 
                    # but here we want a Tree, so let's pop or copy. Let's copy for simplicity.
                    # Warning: Copying nodes is tricky with parent pointers. 
                    # Simpler for demo: Just instantiate new Leafs if needed or use simple logic.
                    # Let's just use what's in pool and remove them to build a tree.
                    c1 = pool.pop(max(idx1, idx2))
                    c2 = pool.pop(min(idx1, idx2))
                    
                    if {c1.label, c2.label} == {'A', 'B'}:
                        pool.extend([c1, c2]) # Invalid, return
                        print(f"Skipped Branch({c1.label}, {c2.label}): Mutually Exclusive")
                        continue
                        
                    b = Branch(c1, c2)
                    if b.propagate():
                        print(f"Created {b}")
                        pool.append(b)
                    else:
                        pool.extend([c1, c2])
                
                else:
                    if not pool: continue
                    c = pool.pop(random.randint(0, len(pool)-1))
                    
                    new_node = None
                    if op == 'A2B' and c.label in ['A', 'O']: new_node = A2B(c)
                    elif op == 'B2A' and c.label in ['B', 'O']: new_node = B2A(c)
                    elif op == 'AB2O' and c.label in ['A', 'B']: new_node = AB2O(c)
                    
                    if new_node and new_node.propagate():
                        print(f"Created {new_node}")
                        pool.append(new_node)
                    else:
                        pool.append(c) # Return on fail
                        
            except Exception:
                continue

        print("Iteration limit reached.")
        return None

class TreeEditor:
    @staticmethod
    def mask_node(node: Node) -> Empty:
        print(f"\n[Masking] Replacing {node} with Empty...")
        parent = node.parent
        new_empty = Empty() # {A,B,O}
        
        if parent:
            idx = parent.children.index(node)
            parent.children[idx] = new_empty
            new_empty.parent = parent # This triggers get_allowed_labels check inside set_parent usually, 
                                      # but Node.set_parent calls propagate. 
                                      # Let's do it manually to show the logic clearly.
            
            # 1. Ask parent what is allowed (Context Awareness)
            allowed = parent.get_allowed_labels_for_child(new_empty)
            print(f"   -> Constraint from Parent ({parent.__class__.__name__}): Allowed = {allowed}")
            
            # 2. Apply intersection
            new_empty.candidates = new_empty.candidates.intersection(allowed)
            
            # 3. Propagate to ensure consistency
            new_empty.propagate()
            
        print(f"   -> Result: {parent}")
        return new_empty

# ==========================================
# 3. Main Execution
# ==========================================

if __name__ == "__main__":
    print("==========================================")
    print("SCENARIO 1: Top-Down Generation (Root=A)")
    print("==========================================")
    td = TopDownEngine(root_label='A')
    tree1 = td.run()

    print("\n==========================================")
    print("SCENARIO 2: Bottom-Up Construction (Target=O)")
    print("==========================================")
    bu = BottomUpEngine()
    tree2 = bu.run(target_label='O')

    print("\n==========================================")
    print("SCENARIO 3: Verification of 'Masking/Resampling'")
    print("==========================================")
    # 构造特定的 Case: AB2O(Leaf(A)) -> 替换 Leaf(A) -> 应该允许 {A, B}
    
    print("1. Constructing manual tree: AB2O(Leaf(A))...")
    l_a = Leaf('A')
    root = AB2O(l_a)
    root.propagate()
    print(f"   Tree: {root}")
    print(f"   Root candidates: {root.candidates}")
    print(f"   Child candidates: {root.children[0].candidates}")
    
    # Masking
    TreeEditor.mask_node(root.children[0])
    # 预期：AB2O(O)[Empty({A,B})]
    # 解释：AB2O 要求 child 是 A 或 B 才能输出 O。
    # 如果 child 是 O，AB2O 会变成 B (参见 A2B/B2A逻辑) 或者 invalid (AB2O 定义输入 A/B 输出 O)
    # 根据代码 AB2O._map: if c in [A,B] return O. else return None.
    # 所以如果 root 锁死为 O，child 只能是 A 或 B。