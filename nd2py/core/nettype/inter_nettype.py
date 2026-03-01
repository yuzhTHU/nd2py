# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import warnings
import itertools
from ..base_visitor import Visitor, yield_nothing
from .nettype_mixin import ALL_NETTYPES

class InferNettype(Visitor):
    """
    基于无栈遍历 (Stackless Traversal) 的类型推导访问器，
    采用两遍扫描法（自底向上 + 自顶向下），自动处理树上的节点类型约束传播。
    """
    def __call__(self, root: 'Symbol') -> None:
        # 需要从根节点开始传播，确保整个树都被覆盖
        if root.parent is not None:
            while root.parent is not None:
                root = root.parent
        # 首先重置所有节点的 candidates 到初始状态
        super().__call__(root, "reset")
        # 迭代进行自底向上和自顶向下的传播直到收敛 (理论上树结构应该 1 次迭代就收敛, 但为了安全起见设置了最大迭代次数 = 10)
        for iteration in range(10):
            # 自底向上通过叶子结点的类型推导父节点的类型约束
            changed_bottom_up = super().__call__(root, "bottom_up")
            # 自顶向下通过父节点的类型约束收紧子节点的类型候选集
            changed_top_down = super().__call__(root, "top_down")
            # 检测是否收敛
            if not changed_bottom_up and not changed_top_down:
                break
            # 对于树结构，理论上一次 Bottom-UP + Top-Down 就应该收敛，因此在 iteration > 0 时发出警告。
            if iteration > 0:
                warnings.warn(
                    f"NetType inference required more than 1 full pass to converge (iteration={iteration}). "
                    "This implies either a DAG topological structure or a non-deterministic mapping."
                )
        else:
            raise RuntimeError("Inference loop exceeded maximum iterations (10 times). Conflict in cyclic graph?")

    def generic_visit(self, node: 'Symbol', action: str):
        yield from yield_nothing() # 保证这是一个生成器函数
        changed = False

        if action == "reset":
            # 自顶向下清空所有节点的 _possible_nettypes 到初始状态
            # 这里不给 node.possible_nettypes 赋值是担心重复触发 infer_nettype 导致奇怪的问题，下同
            node._possible_nettypes = node._assigned_nettypes.copy() if node._assigned_nettypes else ALL_NETTYPES.copy()
            for child in node.operands:
                yield child, ("reset",), {}
            changed = True
        elif action == "bottom_up":
            # [自底向上推导逻辑 (后续遍历)] 递归获取所有子节点的结果, 收缩自己的类型
            if len(node.operands) > 0:
                for child in node.operands:
                    changed |= (yield child, ("bottom_up",), {})
                possible_nettypes = set()
                for children_nettype in itertools.product(*[child.possible_nettypes for child in node.operands]):
                    if (nettype := node.map_nettype(*children_nettype)) is not None: # 合法的映射
                        possible_nettypes.add(nettype)
                possible_nettypes &= node.possible_nettypes # 只能收缩不能扩张
                if not possible_nettypes:
                    raise ValueError(f"Type Inference Conflict at {node}: No valid combination for children.")
                if node.possible_nettypes != possible_nettypes:
                    node._possible_nettypes = possible_nettypes
                    changed = True
        elif action == "top_down":
            # [(自顶向下推导逻辑) 先序遍历] 基于父节点约束向下收缩子节点的类型
            if len(node.operands) > 0:
                possible_children_nettypes = []
                for children_nettype in itertools.product(*[child.possible_nettypes for child in node.operands]):
                    if node.map_nettype(*children_nettype) in node.possible_nettypes:
                        possible_children_nettypes.append(children_nettype)           
                for idx, child in enumerate(node.operands):
                    possible_nettypes = set()
                    for children_nettype in possible_children_nettypes:
                        possible_nettypes.add(children_nettype[idx])
                    possible_nettypes &= child.possible_nettypes # 只能收缩不能扩张
                    if not possible_nettypes:
                        raise ValueError(f"Type Inference Conflict at child {child} of {node}.")
                    if child.possible_nettypes != possible_nettypes:
                        child._possible_nettypes = possible_nettypes
                        changed = True
                    changed |= (yield child, ("top_down",), {})
        else:
            raise ValueError(f"Unknown action: {action}")
        return changed
