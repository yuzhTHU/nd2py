from .abo_tree import *

class TopDownEngine:
    def __init__(self, root_label: Label):
        self.root = Empty()
        self.root.candidates = {root_label}
        
    def get_empty_nodes(self, node: Node, result: List[Empty]):
        if isinstance(node, Empty):
            result.append(node)
        for child in node.children:
            self.get_empty_nodes(child, result)

    def solve(self):
        step = 0
        while True:
            empties = []
            self.get_empty_nodes(self.root, empties)
            if not empties:
                break
            
            print(f"--- Step {step}: Remaining Empties: {len(empties)} ---")
            
            # 1. 随机选择一个 Empty
            target_empty = random.choice(empties)
            
            # 2. 尝试所有可能的填充物
            # 我们不仅要看 label 是否匹配，还要看放入后 propagate 是否成功
            
            possible_moves = []
            
            # 尝试填充 Leaf
            for l in ['A', 'B', 'O']:
                # 只有当 target_empty 允许该 label 时才尝试
                if l in target_empty.candidates:
                    possible_moves.append(lambda p=target_empty, lbl=l: self.apply_leaf(p, lbl))
            
            # 尝试填充 Branch
            # Branch 本身对 Label 没有硬性限制（取决于 child），但在初始状态，它必须兼容 target_empty.candidates
            # 对于 Branch, A2B 等，我们先放入全是 Empty 的结构，让 propagate 去收缩
            possible_moves.append(lambda p=target_empty: self.apply_branch(p))
            
            # 尝试填充 A2B (如果 Empty 允许 B)
            if 'B' in target_empty.candidates:
                possible_moves.append(lambda p=target_empty: self.apply_transformer(p, A2B))
            # 尝试填充 B2A (如果 Empty 允许 A)
            if 'A' in target_empty.candidates:
                possible_moves.append(lambda p=target_empty: self.apply_transformer(p, B2A))
            # 尝试填充 AB2O (如果 Empty 允许 O)
            if 'O' in target_empty.candidates:
                possible_moves.append(lambda p=target_empty: self.apply_transformer(p, AB2O))
            
            # 3. 随机选择并执行一个 Move
            # 注意：实际应用中可能需要回溯（Deepcopy tree），这里做简化：假设总能找到合法路径
            # 为了演示，我们尝试直到成功
            random.shuffle(possible_moves)
            success = False
            for move in possible_moves:
                # 这里的逻辑比较简单，实际生产环境应该 clone 整个树来 try
                # 这里我们假设只要 candidates 允许，基本就能填入，具体的冲突由 propagate 处理
                # 如果 propagate 返回 False，我们需要撤销（这里略去复杂的撤销逻辑，仅展示流程）
                try:
                    move()
                    success = True
                    break
                except Exception as e:
                    print(f"Move failed: {e}")
                    continue
            
            if not success:
                print("Dead end reached (Constraints too tight).")
                return

            step += 1
            print(f"Current Root State: {self.root}")

    # === Actions ===
    
    def replace_node(self, target: Empty, new_node: Node):
        # 继承 target 的约束
        new_node.candidates = new_node.candidates.intersection(target.candidates)
        if not new_node.candidates:
            raise ValueError("Candidate mismatch")
            
        parent = target.parent
        new_node.parent = parent
        
        if parent:
            idx = parent.children.index(target)
            parent.children[idx] = new_node
        else:
            self.root = new_node
            
        # 触发约束传播！这是最关键的一步
        if not new_node.propagate():
            # 实际应当回滚
            raise ValueError("Propagation violation")

    def apply_leaf(self, target: Empty, label: Label):
        print(f"Applying Leaf({label}) to {target}")
        self.replace_node(target, Leaf(label))

    def apply_branch(self, target: Empty):
        print(f"Applying Branch to {target}")
        # 创建 Branch，带着两个 Empty Child
        b = Branch(Empty(), Empty())
        self.replace_node(target, b)

    def apply_transformer(self, target: Empty, cls):
        print(f"Applying {cls.__name__} to {target}")
        t = cls(Empty())
        self.replace_node(target, t)