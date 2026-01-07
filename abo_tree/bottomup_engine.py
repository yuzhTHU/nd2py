from .abo_tree import *

class BottomUpEngine:
    def generate(self, target_label: Label, max_iter=100) -> Optional[Node]:
        pool: List[Node] = []
        
        # 初始池
        for _ in range(5):
            pool.extend([Leaf('A'), Leaf('B'), Leaf('O')])
            
        for i in range(max_iter):
            # 检查是否有符合要求的根节点
            matches = [n for n in pool if n.label == target_label]
            if matches and i > 5: # 稍微迭代几次增加复杂度
                return random.choice(matches)
            
            # 随机选择操作符
            op = random.choice(['Branch', 'A2B', 'B2A', 'AB2O'])
            
            try:
                if op == 'Branch':
                    # 需要选两个
                    if len(pool) < 2: continue
                    c1 = pool.pop(random.randint(0, len(pool)-1))
                    c2 = pool.pop(random.randint(0, len(pool)-1))
                    
                    # 检查 Branch 约束 (不能 A+B)
                    l1, l2 = c1.label, c2.label # Bottom-up 时 label 应该是确定的
                    if {l1, l2} == {'A', 'B'}:
                        # 放回去，无效组合
                        pool.append(c1)
                        pool.append(c2)
                        continue
                        
                    node = Branch(c1, c2)
                    # 计算 label (Branch 初始化时逻辑未自动计算确定的 candidates，这里需手动触发一下或改进 Branch)
                    # 实际上 Branch 的 update_self_from_children 会搞定
                    if node.propagate():
                        pool.append(node)
                    
                else:
                    # 单子节点类
                    if len(pool) < 1: continue
                    c1 = pool.pop(random.randint(0, len(pool)-1))
                    
                    new_node = None
                    if op == 'A2B' and c1.label in ['A', 'O']:
                        new_node = A2B(c1)
                    elif op == 'B2A' and c1.label in ['B', 'O']:
                        new_node = B2A(c1)
                    elif op == 'AB2O' and c1.label in ['A', 'B']:
                        new_node = AB2O(c1)
                    
                    if new_node and new_node.propagate():
                        pool.append(new_node)
                    else:
                        pool.append(c1) # 只有不符合逻辑才放回
                        
            except Exception:
                continue
                
        return None