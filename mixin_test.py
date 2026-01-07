import unittest
import warnings
from typing import Set, List, Optional, Literal
from copy import deepcopy

# =============================================================================
# 1. 核心引擎实现 (Engine Implementation) - [最终修复版]
# =============================================================================

NetType = Literal["node", "edge", "scalar"]
ALL_NETTYPES: Set[NetType] = {"node", "edge", "scalar"}

class NetTypeMixin:
    parent: Optional['Symbol']
    operands: List['Symbol']

    def __init__(self, fixed_nettype: Optional[NetType] = None):
        self._initial_candidates: Set[NetType] = (
            {fixed_nettype} if fixed_nettype else ALL_NETTYPES.copy()
        )
        self._candidates: Set[NetType] = self._initial_candidates.copy()

    @property
    def candidates(self) -> Set[NetType]:
        return self._candidates

    @candidates.setter
    def candidates(self, val: Set[NetType]):
        if self._candidates == val: return
        if not val:
            raise ValueError(f"NetType Conflict: {self} candidates became empty.")
        
        self._candidates = val
        # [FIX 1: Force Push] Setter 触发的改变，必须强制通知邻居
        self.propagate(force_push=True)

    def propagate(self, force_push=False) -> bool:
        """
        force_push: 跳过变更检测，强制通知 Parent 和 Children
        """
        old_candidates = self._candidates.copy()

        # Step 1: Self-Check (Bottom-up)
        if not self._update_self_from_children():
            return False 

        # Step 2: Push (Top-down / Sideways)
        # 触发条件：要么是 force_push (setter触发)，要么是自检导致了收缩
        if force_push or (self._candidates != old_candidates):
            if self.parent:
                self.parent.propagate() # Parent 会回头读新值
            for child in self.operands:
                self._push_constraint_to_child(child)
        return True

    def set_parent_with_constraint(self, parent: 'Symbol'):
        self.parent = parent
        allowed = parent.get_allowed_nettypes_for_child(self)
        self.candidates = self.candidates.intersection(allowed)

    def _push_constraint_to_child(self, child: 'Symbol'):
        allowed = self.get_allowed_nettypes_for_child(child)
        new_child_cand = child.candidates.intersection(allowed)
        if new_child_cand != child.candidates:
            child.candidates = new_child_cand

    def _update_self_from_children(self) -> bool: return True
    def get_allowed_nettypes_for_child(self, child: 'Symbol') -> Set[NetType]: return ALL_NETTYPES.copy()


class TreeEditor:
    @staticmethod
    def replace_operand(parent: 'Symbol', index: int, new_child: 'Symbol'):
        # 1. 临时占位 (断开连接)
        temp = Empty() 
        parent.operands[index] = temp 
        
        # 2. 祖先链松弛
        TreeEditor._relax_ancestry(parent)
        
        # 3. 正式连接
        parent.set_operand(index, new_child)

    @staticmethod
    def _relax_ancestry(node: 'Symbol'):
        path = []
        curr = node
        while curr:
            path.append(curr)
            curr = curr.parent
        
        # 自顶向下重置
        for ancestor in reversed(path):
            # [FIX 2: Direct Set] 绕过 setter，纯粹洗白数据
            ancestor._candidates = ancestor._initial_candidates.copy()
            
            # 吸收父节点约束
            if ancestor.parent:
                allowed = ancestor.parent.get_allowed_nettypes_for_child(ancestor)
                ancestor._candidates = ancestor._candidates.intersection(allowed)
            
            # [FIX 3: Recursive Reset] 关键！把兄弟节点的幽灵约束也洗掉
            for child in ancestor.operands:
                TreeEditor._recursive_reset_to_initial(child)
            
            # [FIX 4] 手动触发传播，重新建立正确的约束关系
            ancestor.propagate(force_push=True)

    @staticmethod
    def _recursive_reset_to_initial(node: 'Symbol'):
        # 同样绕过 setter
        node._candidates = node._initial_candidates.copy()
        for child in node.operands:
            TreeEditor._recursive_reset_to_initial(child)

# =============================================================================
# 2. 符号类定义 (Symbol Definitions) - [已修复初始化顺序]
# =============================================================================

class Symbol(NetTypeConstraintMixin):
    n_operands = 0
    def __init__(self, *operands, nettype: NetType = None):
        NetTypeConstraintMixin.__init__(self, fixed_nettype=nettype)
        self.parent = None
        
        # 清洗
        ops = list(operands)
        if len(ops) == 0 and self.n_operands > 0:
            ops = [Empty() for _ in range(self.n_operands)]
            
        # [FIX 5: Init Order] 先物理填充，再逻辑连接
        self.operands = []
        for op in ops:
            if op.parent: op = deepcopy(op)
            self.operands.append(op)
            
        for op in self.operands:
            op.set_parent_with_constraint(self)
            
        if not self.propagate():
             raise ValueError(f"Init Conflict in {type(self).__name__}")

    def set_operand(self, index: int, child: 'Symbol'):
        if child.parent: child = deepcopy(child)
        self.operands[index] = child
        child.set_parent_with_constraint(self)

    def __repr__(self):
        c_str = f"{{{','.join(sorted(self.candidates))}}}"
        if len(self.candidates) == 1: c_str = list(self.candidates)[0]
        if not self.operands: return f"{type(self).__name__}({c_str})"
        return f"{type(self).__name__}({c_str}, {self.operands})"

class Empty(Symbol):
    pass

class Leaf(Symbol):
    def __init__(self, nettype: NetType):
        super().__init__(nettype=nettype)

class Add(Symbol):
    n_operands = 2
    
    def _update_self_from_children(self) -> bool:
        c0 = self.operands[0].candidates
        c1 = self.operands[1].candidates
        common = c0.intersection(c1)
        self.candidates = self.candidates.intersection(common)
        return True

    def get_allowed_nettypes_for_child(self, child: Symbol) -> Set[NetType]:
        sibling = self.operands[1] if child is self.operands[0] else self.operands[0]
        return self.candidates.intersection(sibling.candidates)

class Sour(Symbol):
    n_operands = 1
    def __init__(self, *args, **kwargs):
        super().__init__(*args, nettype='edge', **kwargs)
        
    def _update_self_from_children(self) -> bool:
        if 'node' not in self.operands[0].candidates:
            self.candidates = set() 
        return True

    def get_allowed_nettypes_for_child(self, child: Symbol) -> Set[NetType]:
        return {'node'}

class Aggr(Symbol):
    n_operands = 1
    def __init__(self, *args, **kwargs):
        super().__init__(*args, nettype='node', **kwargs)

    def _update_self_from_children(self) -> bool:
        if 'edge' not in self.operands[0].candidates:
            self.candidates = set()
        return True
        
    def get_allowed_nettypes_for_child(self, child: Symbol) -> Set[NetType]:
        return {'edge'}

# =============================================================================
# 3. 测试用例 (Test Cases)
# =============================================================================

class TestNetTypeConstraints(unittest.TestCase):

    def test_01_basic_inference_add(self):
        print("\n--- Test 01: Add(Node, Node) -> Node ---")
        n1 = Leaf(nettype='node')
        n2 = Leaf(nettype='node')
        add = Add(n1, n2)
        self.assertEqual(add.candidates, {'node'})
        print(f"Result: {add}")

    def test_02_conflict_detection(self):
        print("\n--- Test 02: Add(Node, Edge) -> Conflict ---")
        n = Leaf(nettype='node')
        e = Leaf(nettype='edge')
        with self.assertRaisesRegex(ValueError, "NetType Conflict"):
            Add(n, e)
        print("Conflict correctly caught.")

    def test_03_top_down_sour(self):
        print("\n--- Test 03: Sour(Empty) -> Empty must be Node ---")
        empty = Empty()
        sour = Sour(empty)
        self.assertEqual(sour.candidates, {'edge'})
        self.assertEqual(empty.candidates, {'node'})
        print(f"Result: {sour}")

    def test_04_circular_consistency(self):
        print("\n--- Test 04: Circular Consistency in Add ---")
        add = Add(Empty(), Empty()) 
        # 外部强制设为 node
        add.candidates = {'node'}
        self.assertEqual(add.operands[0].candidates, {'node'})
        self.assertEqual(add.operands[1].candidates, {'node'})
        print(f"Result: {add}")

    def test_05_tree_editor_relaxation(self):
        print("\n--- Test 05: Tree Editor Relaxation ---")
        l_node = Leaf(nettype='node')
        r_empty = Empty()
        root = Add(l_node, r_empty)
        
        print(f"Before: {root}")
        self.assertEqual(root.candidates, {'node'})
        self.assertEqual(r_empty.candidates, {'node'})
        
        # 替换：移除左边 Node
        new_left = Empty()
        TreeEditor.replace_operand(root, 0, new_left)
        
        print(f"After : {root}")
        # Root 和孩子都应该恢复自由
        self.assertEqual(root.candidates, ALL_NETTYPES)
        self.assertEqual(r_empty.candidates, ALL_NETTYPES)
        self.assertEqual(new_left.candidates, ALL_NETTYPES)

    def test_06_complex_chain_relaxation(self):
        print("\n--- Test 06: Complex Chain Relaxation ---")
        l_node = Leaf('node')
        r_empty = Empty()
        add = Add(l_node, r_empty) 
        sour = Sour(add)           
        
        print(f"Chain: {sour}")
        
        new_empty = Empty()
        TreeEditor.replace_operand(add, 0, new_empty)
        
        print(f"After Replace: {sour}")
        # Add 依然必须是 Node (受 Sour 限制)
        self.assertEqual(add.candidates, {'node'})
        
        new_root_child = Empty()
        TreeEditor.replace_operand(sour, 0, new_root_child)
        
        print(f"After Root Replace: {sour}")
        self.assertEqual(new_root_child.candidates, {'node'})

    def test_07(self):
        print("\n--- Test 07 ---")
        l_node = Empty()
        l_node._candidates = {'node', 'scalar'}
        r_node = Empty()
        r_node._candidates = {'scalar', 'edge'}
        add = Add(l_node, r_node) 
        
        print(f"Chain: {sour}")
        
        new_empty = Empty()
        TreeEditor.replace_operand(add, 0, new_empty)
        
        print(f"After Replace: {sour}")
        # Add 依然必须是 Node (受 Sour 限制)
        self.assertEqual(add.candidates, {'node'})
        
        new_root_child = Empty()
        TreeEditor.replace_operand(sour, 0, new_root_child)
        
        print(f"After Root Replace: {sour}")
        self.assertEqual(new_root_child.candidates, {'node'})



if __name__ == '__main__':
    unittest.main()