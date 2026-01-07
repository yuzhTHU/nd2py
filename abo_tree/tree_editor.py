from .abo_tree import *

class TreeEditor:
    @staticmethod
    def mask_node(node_to_replace: Node) -> Empty:
        """
        将树中的某个节点替换为 Empty，并自动最大化该 Empty 的可能性。
        """
        parent = node_to_replace.parent
        new_empty = Empty() # 初始 {A, B, O}
        
        if not parent:
            # 如果替换的是根节点，它就是完全自由的（除非有额外的 root_label 约束，但在 node 层面它自由）
            # 或者我们可以继承原 root 的 label？通常 mask 意味着我想改变它，所以保持全集
            return new_empty
        
        # 1. 在 Parent 中替换引用
        idx = parent.children.index(node_to_replace)
        parent.children[idx] = new_empty
        new_empty.parent = parent
        
        # 2. 计算 Context Constraint (Context = Parent + Siblings)
        # 这一步实现了 "区分子代约束和祖先约束"。
        # 我们忽略了 node_to_replace 带来的约束，只看 parent 允许什么。
        allowed_by_context = parent.get_allowed_labels_for_child(new_empty)
        
        # 3. 更新 Empty
        new_empty.candidates = new_empty.candidates.intersection(allowed_by_context)
        
        # 4. 触发传播 (防止上下文有潜在冲突，尽管在 Mask 步骤通常是放宽约束)
        # 注意：这里我们不需要从 Root 开始重算，因为我们假设 Parent 的 candidates 
        # 是由 Parent 的 Ancestors 固定的（Top-down constraint）。
        new_empty.propagate()
        
        return new_empty

    @staticmethod
    def fill_empty(empty_node: Empty, engine):
        """
        使用之前的 TopDownEngine 逻辑仅填充这个子树
        """
        # 这里复用 TopDownEngine 的单步填充逻辑
        # ... (代码省略，逻辑同上文的 engine.solve 循环)
        pass