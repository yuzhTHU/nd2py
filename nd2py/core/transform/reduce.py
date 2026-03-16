# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import pickle
import inspect
import hashlib
import logging
import numpy as np
from pathlib import Path
from functools import reduce
from typing import List, Tuple, Dict, TYPE_CHECKING
from tqdm import tqdm
from .fold_constant import FoldConstant
if TYPE_CHECKING:
    from ..symbols import *

_logger = logging.getLogger(f'nd2py.{__name__}')

# 缓存文件路径
_CACHE_DIR = Path("~/.cache/nd2py/reduce")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ReduceRule:
    def __init__(self, source: Symbol, target: Symbol):
        self.source = source
        self.target = target
    
    def __repr__(self):
        return f'{self.source} => {self.target}'
    
    def __str__(self):
        return f'{self.source} => {self.target}'


class Reduce:
    def __init__(
            self, 
            n_variables=4,
            constants=[0, 1, -1, np.pi, np.e],
            binary=None,
            unary=None,
            max_online_iterations: int = 10, 
            load_cache: bool = True,
            num_anchors = 100,
        ):
        from .. import symbols as nd
        self.variables = [nd.Variable(f'x{n}') for n in range(n_variables)]
        self.constants = [nd.Number(c) for c in constants]
        self.leafs = self.variables + self.constants
        self.binary = binary or set(v for v in vars(nd).values() if inspect.isclass(v) and issubclass(v, nd.Symbol) and v.n_operands == 2)
        self.unary = unary or set(v for v in vars(nd).values() if inspect.isclass(v) and issubclass(v, nd.Symbol) and v.n_operands == 1)
        self.max_online_iterations = max_online_iterations

        _rng = np.random.default_rng(42)
        self.anchors = {var.name: _rng.uniform(-5, 5, num_anchors) for var in list(self.variables)}
        self.hash_dict: Dict[int, Tuple[Symbol, np.ndarray]] = {}
        self.reduce_rules: List[ReduceRule] = []  # 化简规则，按模式长度倒序排列 (longest to smallest)
        if load_cache:
            self._load_from_cache() # 初始化时加载缓存
    
    def __hash__(self):
        config = {
            'variables': sorted([v.name for v in self.variables]),
            'constants': sorted([c.value for c in self.constants]),
            'binary': sorted([op.__name__ for op in self.binary]),
            'unary': sorted([op.__name__ for op in self.unary]),
        }
        return int(hashlib.md5(str(config).encode()).hexdigest(), 16)

    @property
    def _cache_path(self) -> Path:
        return Path(_CACHE_DIR) / f"rules_{hash(self)}.pkl"

    def _save_to_cache(self):
        """保存规则到缓存"""
        with open(self._cache_path, 'wb') as f:
            pickle.dump((self.anchors, self.hash_dict, self.reduce_rules), f)

    def _load_from_cache(self):
        """从缓存加载规则"""
        if not self._cache_path.exists():
            return
        with open(self._cache_path, 'rb') as f:
            self.anchors, self.hash_dict, self.reduce_rules = pickle.load(f)

    def prepare_rule_dict(self, l_max: int = 8, force_rebuild: bool = False, save_cache: bool = True, show_progress: bool = True):
        """
        离线阶段：基于 Kruskal 最小生成森林算法的变体发现化简规则并构建 reduce_rules 字典。

        Args:
            l_max: 最大表达式长度
            force_rebuild: 是否强制重新构建（忽略缓存）
            save_cache: 是否将构建结果缓存到文件
            show_progress: 是否显示进度条
        """
        from ..symbols import Number
        from ...generator.eq import Enumerator

        if force_rebuild: # 清空缓存
            self.hash_dict = {}
            self.reduce_rules = []
        enumerator = Enumerator(leafs=self.leafs, binary=self.binary, unary=self.unary)

        # 外层进度条：显示长度进度
        l_start = min(
            max([0] + [len(eq) for bucket in self.hash_dict.values() for eq, val in bucket]),
            max([0] + [len(rule.source) for rule in self.reduce_rules]) ,
        )
        progress_bar = tqdm(
            range(l_start + 1, l_max + 1),
            desc="Length",
            unit="L",
            # ncols=True,
            # leave=False,
            disable=not show_progress
        )

        for L in progress_bar:
            # 内层进度条：显示当前长度内的处理进度
            inner_progress = tqdm(
                enumerator(length=L),
                total=enumerator.estimate_total(L),
                desc=f"Formulas L={L}",
                unit="expr",
                # ncols=True,
                # leave=False,
                disable=not show_progress
            )
            for tau in inner_progress:
                if not any(type(var).__name__ == 'Variable' for var in tau.iter_preorder()):
                    continue # tau 中不含变量，为常量表达式

                val = tau.eval(self.anchors)
                if np.size(val) == 1:
                    val = np.full(next(iter(self.anchors.values())).shape, val)
                if not np.isfinite(val).all():
                    val = np.nan_to_num(val, nan=1.23456789, posinf=2.34567891, neginf=3.45678912)

                if np.isclose(val, val[0], atol=1e-10, rtol=1e-10).all():
                    if   np.isclose(val[0], 1.23456789, atol=1e-10, rtol=1e-10): tau2 = Number(np.nan)
                    elif np.isclose(val[0], 2.34567891, atol=1e-10, rtol=1e-10): tau2 = Number(+np.inf)
                    elif np.isclose(val[0], 3.45678912, atol=1e-10, rtol=1e-10): tau2 = Number(-np.inf)
                    else: tau2 = Number(val[0])
                    self.reduce_rules.append(ReduceRule(tau, tau2))
                else:
                    val_rounded = np.round(val, 8)
                    # 必须使用 ascontiguousarray 保证内存布局一致，否则切片或转置的数组即使数值相同，字节流也不同
                    val_bytes = np.ascontiguousarray(val_rounded).tobytes()
                    # 跨进程稳定的 256 位哈希：彻底消除真实碰撞和 Cache 毒性
                    val_hash = hashlib.sha256(val_bytes).hexdigest()
                    # 字典桶机制 (Hash Bucket) 查找
                    if val_hash not in self.hash_dict:
                        # 如果哈希键完全不存在，初始化一个新的哈希桶（列表）
                        self.hash_dict[val_hash] = [(tau, val)]
                    elif not (candidates := [t for t, v in self.hash_dict[val_hash] if np.isclose(val, v, atol=1e-8, rtol=1e-5).all()]):
                        # 发生了哈希碰撞但二次校验未通过, 将当前新表达式安全地追加到同一个桶中
                        self.hash_dict[val_hash].append((tau, val))
                    elif len(best_tau := min(candidates, key=lambda x: len(x))) < len(tau): # 未来可能可以有更好的 key 挑选 best_tau
                        # 如果最短公式比 tau 更短，说明存在化简规则 tau -> best_tau
                        self.reduce_rules.append(ReduceRule(tau, best_tau))
                    else:
                        # 桶中的最短公式和 tau 一样长，不妨将 tau 也加到桶中
                        self.hash_dict[val_hash].append((tau, val))

                # 更新内层进度条后缀信息
                inner_progress.set_postfix({"rules": len(self.reduce_rules)})

            # 更新外层进度条后缀信息
            progress_bar.set_postfix({"total_rules": len(self.reduce_rules)})

            if save_cache: # 保存到缓存
                self._save_to_cache()

    def __call__(self, root: Symbol) -> Symbol:
        """
        在线阶段：交替运行规则匹配和代数清理，直到收敛。
        """
        current_expr = root.copy()
        for _ in range(self.max_online_iterations):
            _current_expr = current_expr.copy()
            current_expr = self._rule_pass(current_expr)[0]
            current_expr = FoldConstant(fold_fitable=True, fold_constant=True)(current_expr)
            current_expr = reduce(lambda a, b: a + b, current_expr.split_by_add(split_by_sub=True, expand_mul=True, expand_div=True, merge_bias=True))
            if str(_current_expr) == str(_current_expr):
                break # 如果表达式没有改变，说明已收敛
        return current_expr

    def _rule_pass(self, node: Symbol) -> Tuple[Symbol, bool]:
        """
        自顶向下尝试化简，失败则下潜至子节点，然后自底向上重新检查。
        """
        # 尝试带变量的模式匹配 (按长度从大到小)
        for rule in self.reduce_rules:
            if (match := node.match(rule.source)) is not None: 
                # match 捕获了 "rule.source 中变量 -> node 中子树" 的映射
                # 将 rule.target 中变量替换为 match 中捕获的 node 子树
                target = rule.target.copy()
                var_list = [var for var in target.iter_preorder() if type(var).__name__ == 'Variable']
                for var in var_list:
                    target = target.replace(var, match[var.name].copy(), no_warn=True)
                return target, True

        # 自顶向下：如果当前层级化简失败，递归化简子节点
        children_results = []
        for child in node.operands:
            new_child, child_changed = self._rule_pass(child.copy())
            if child_changed:
                children_results.append((child, new_child))
        for child, new_child in children_results:
            node = node.replace(child, new_child, no_warn=True)
        return node, len(children_results) > 0

