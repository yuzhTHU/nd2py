from __future__ import annotations
import warnings
import itertools
from typing import Set
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Set, List, Optional, Literal
from ..base_visitor import Visitor, yield_nothing
from ..context.nettype_inference import nettype_inference
from ...utils import classproperty

if TYPE_CHECKING: # 避免循环引用，仅用于类型检查
    from ..symbols import Symbol

# 定义核心类型
NetType = Literal["node", "edge", "scalar"]
ALL_NETTYPES: Set[NetType] = {"node", "edge", "scalar"}


class NetTypeMixin(ABC):
    """
    Mixin 类：负责维护 Symbol 的 nettype 候选集 (nettypes) 并处理约束传播。
        - 变量 (Variable), 数值 (Number) 和占位符 (Empty) 可以具有特定的 nettype, 表示该对象在网络中的角色 (节点、边或标量)
        - 不同运算符对操作符的 nettype 有不同的要求和限制, 所得输出的 nettype 也不同
        - 被指定为特定 nettype 的对象 (如 Variable(nettype="node")) 成为固定的锚点 (Anchor), 辅助其他对象通过约束传播确定自己的 nettype 候选集合

    宿主类 (Symbol) 必须提供以下属性和方法:
        - self.parent
        - self.operands
        - self.map_nettype(*children_nettypes: NetType) -> Optional[NetType]
    """

    # 通过 Type Hinting 提示宿主类必须有的属性, 不产生任何实际效果, 但方便 IDE 和类型检查器识别
    parent: Optional[Symbol]
    operands: List[Symbol]

    @classmethod
    @abstractmethod
    def map_nettype(cls, *children_nettypes: NetType) -> Optional[NetType]:
        """
        根据子节点 nettypes 推导自身 nettype, 返回 None 表示发现逻辑冲突.
        需要由具体的 Symbol 子类实现自己的映射逻辑
        """
        return None

    def __init__(self, nettype: Optional[NetType] = None):
        if isinstance(nettype, str) and nettype in ALL_NETTYPES: nettype = {nettype}
        # nettype 可以是 None 或 Set[NetType]
        # 如果用户提供了 nettype 作为硬约束, 将其用作限制其它节点的锚点
        self._assigned_nettypes: Set[NetType] = nettype.copy() if nettype else None
        # 此对象可取的 nettype 集合
        self._possible_nettypes: Set[NetType] = nettype.copy() if nettype else ALL_NETTYPES.copy()

    @property
    def possible_nettypes(self) -> Set[NetType]:
        """ 返回此对象可取的 nettype 集合 """
        return self._possible_nettypes
    
    @property
    def nettype(self) -> NetType|None:
        """ 如果此对象可取的 nettype 唯一，则返回该 nettype，否则返回 None 表示不确定。"""
        return next(iter(self._possible_nettypes)) if len(self._possible_nettypes) == 1 else None

    @possible_nettypes.setter
    def possible_nettypes(self, val: Set[NetType]):
        raise AttributeError("Cannot set possible_nettypes directly. Use nettype setter to set hard constraints or infer_nettype() to trigger inference.")
    
    @nettype.setter
    def nettype(self, val: NetType | Set[NetType] | None):
        val = {val} if isinstance(val, str) and val in ALL_NETTYPES else val
        # 性能优化：值未变则跳过
        if self._assigned_nettypes == val: return
        # 冲突检测
        if isinstance(val, set) and len(val) == 0: 
            raise ValueError(f"NetType Conflict: Possible nettypes became empty for {self}.")
        # 赋值并触发全树的 nettype 更新
        self._assigned_nettypes = val
        self.infer_nettype()

    def infer_nettype(self):
        """ 根据 _assigned_nettypes 推断整个符号树的 _possible_nettypes """
        if nettype_inference():
            infer_nettype = InferNettype()
            infer_nettype(self)

    @classmethod
    def get_nettype_range(cls) -> Set[NetType]:
        """ 获取此节点可能产生的所有 nettype 值域，并在首次调用时缓存到类属性中。 """
        if "_nettype_range" not in cls.__dict__: # cls.__dict__ 限定了只检查当前类自身的名称空间, 避免继承父类的 _nettype_range 标记
            possible_range = set()
            for combo in itertools.product(ALL_NETTYPES, repeat=cls.n_operands):
                if (res := cls.map_nettype(*combo)) is not None:
                    possible_range.add(res)
            cls._nettype_range = possible_range
        return cls._nettype_range
    
    @classproperty
    def nettype_range(cls) -> Set[NetType]:
        """ 获取此节点可能产生的所有 nettype 值域，并在首次调用时缓存到类属性中。 """
        return cls.get_nettype_range()


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
