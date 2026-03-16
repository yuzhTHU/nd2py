# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import pickle
import inspect
import hashlib
import logging
import numpy as np
from pathlib import Path
from itertools import islice
from functools import reduce
from collections import defaultdict
from joblib import Parallel, delayed
from platformdirs import user_cache_dir
from typing import List, Tuple, Dict, Optional, NamedTuple, TYPE_CHECKING
from tqdm import tqdm
from .fold_constant import FoldConstant
from ..symbols import Number
from ...generator.eq import Enumerator
if TYPE_CHECKING:
    from ..symbols import *

_logger = logging.getLogger(f'nd2py.{__name__}')

# 缓存文件路径
_CACHE_DIR = Path('.cache/reduce')
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
        self.hash_dict: Dict[int, Tuple[Symbol, np.ndarray]] = defaultdict(list)
        self.reduce_rules: List[ReduceRule] = []  # 化简规则，按模式长度倒序排列 (longest to smallest)
        if load_cache:
            self._load_from_cache()  # 初始化时加载缓存

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
        _logger.info(f'Save Cache to {self._cache_path}')

    def _load_from_cache(self):
        """从缓存加载规则"""
        if not self._cache_path.exists():
            _logger.warning('No Cache Founded!')
            return
        with open(self._cache_path, 'rb') as f:
            self.anchors, self.hash_dict, self.reduce_rules = pickle.load(f)
        _logger.info(f'Load Cache from {self._cache_path}')

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
                break  # 如果表达式没有改变，说明已收敛
        return current_expr

    def prepare_rule_dict(
        self,
        l_max: int = 8,
        force_rebuild: bool = False,
        save_cache: bool = True,
        show_progress: bool = True,
    ):
        """
        离线阶段：基于 Kruskal 最小生成森林算法的变体发现化简规则并构建 reduce_rules 字典。

        这是单进程版本，按顺序处理每个表达式。

        Args:
            l_max: 最大表达式长度
            force_rebuild: 是否强制重新构建（忽略缓存）
            save_cache: 是否将构建结果缓存到文件
            show_progress: 是否显示进度条
        """
        if force_rebuild:
            self.hash_dict = defaultdict(list)
            self.reduce_rules = []
        enumerator = Enumerator(leafs=self.leafs, binary=self.binary, unary=self.unary)
        l_start = self.get_start_length(self.hash_dict, self.reduce_rules)
        progress_bar = tqdm(
            range(l_start, l_max + 1),
            desc="Length",
            unit="L",
            disable=not show_progress,
        )
        for L in progress_bar:
            inner_progress = tqdm(
                enumerator(length=L),
                total=enumerator.estimate_total(L),
                desc=f"Formulas L={L}",
                unit="expr",
                disable=not show_progress,
            )
            loader = (tau for tau in inner_progress if self.has_variable(tau))
            for tau in loader:
                rule, (tau, val, val_hash) = self.process_tau(tau, self.anchors, self.hash_dict)
                if rule is None: 
                    self.reduce_rules.append(rule)
                else: 
                    self.hash_dict[val_hash].append((tau, val))
                inner_progress.set_postfix({"rules": len(self.reduce_rules)})
            inner_progress.close()
            progress_bar.set_postfix({"total_rules": len(self.reduce_rules)})
            if save_cache:
                self._save_to_cache()

    def prepare_rule_dict_parallel(
        self,
        l_max: int = 8,
        n_jobs: int = -1,
        batch_size: int = 10000,
        force_rebuild: bool = False,
        save_cache: bool = True,
        show_progress: bool = True,
    ):
        """
        离线阶段：基于 Kruskal 最小生成森林算法的变体发现化简规则并构建 reduce_rules 字典。

        这是多进程版本，使用 joblib.Parallel 加速同一长度内的表达式处理。

        Args:
            l_max: 最大表达式长度
            n_jobs: 并行进程数，-1 表示使用所有 CPU 核心
            force_rebuild: 是否强制重新构建（忽略缓存）
            save_cache: 是否将构建结果缓存到文件
            show_progress: 是否显示进度条
        """
        if force_rebuild:
            self.hash_dict = defaultdict(list)
            self.reduce_rules = []
        enumerator = Enumerator(leafs=self.leafs, binary=self.binary, unary=self.unary)
        l_start = self.get_start_length(self.hash_dict, self.reduce_rules)
        progress_bar = tqdm(
            range(l_start, l_max + 1),
            desc="Length",
            unit="L",
            disable=not show_progress,
        )
        for L in progress_bar:
            inner_progress = tqdm(
                enumerator(length=L),
                total=enumerator.estimate_total(length=L),
                desc=f"Formulas L={L}",
                unit="expr",
                disable=not show_progress,
            )
            loader = (tau for tau in inner_progress if self.has_variable(tau))
            workers = Parallel(n_jobs=n_jobs, verbose=0, return_as="generator")
            saved_reduce_rules = []
            saved_hash_dict = defaultdict(list)
            if batch_size is None:
                tasks = (delayed(self.process_tau)(tau, self.anchors, self.hash_dict) for tau in loader)
                for rule, (tau, val, val_hash) in workers(tasks):
                    if rule is not None: 
                        saved_reduce_rules.append(rule)
                    else: 
                        saved_hash_dict[val_hash].append((tau, val))
                    inner_progress.update(1)
                    inner_progress.set_postfix({"rules": len(self.reduce_rules)})
            else:
                batch_loader = iter(lambda: list(islice(loader, batch_size)), []) # iter(..., []) 会不断调用前面的 lambda 函数，直到它返回一个空列表 []（即生成器耗尽）时停止。
                tasks = (delayed(self.process_tau_batch)(tau_list, self.anchors, self.hash_dict) for tau_list in batch_loader)
                for results in workers(tasks):
                    for rule, (tau, val, val_hash) in results:
                        if rule is not None:
                            saved_reduce_rules.append(rule)
                        else: 
                            saved_hash_dict[val_hash].append((tau, val))
                        inner_progress.update(1)
                        inner_progress.set_postfix({"rules": len(self.reduce_rules)})
            inner_progress.close()
            self.reduce_rules.extend(saved_reduce_rules)
            for key, values in saved_hash_dict:
                self.hash_dict[key].extend(values)
            progress_bar.set_postfix({"total_rules": len(self.reduce_rules)})
            if save_cache:
                self._save_to_cache()

    @staticmethod
    def has_variable(tau: 'Symbol') -> bool:
        """检查表达式是否包含变量"""
        return any(type(var).__name__ == 'Variable' for var in tau.iter_preorder())

    @staticmethod
    def get_start_length(hash_dict: Dict, reduce_rules: List[ReduceRule]) -> int:
        """获取应该从哪个长度开始处理（跳过已缓存的）"""
        l1 = max([0] + [len(eq) for bucket in hash_dict.values() for eq, val in bucket])
        l2 = max([0] + [len(rule.source) for rule in reduce_rules])
        return min(l1, l2) + 1

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

    @staticmethod
    def get_array_hash(val, n=8):
        val_rounded = np.round(val, n)
        val_bytes = np.ascontiguousarray(val_rounded).tobytes()
        val_hash = hashlib.sha256(val_bytes).hexdigest()
        return val_hash

    @staticmethod
    def process_tau(tau: 'Symbol', anchors: Dict[str, np.ndarray], hash_dict: Dict[str, List[Tuple['Symbol', np.ndarray]]]):
        val = tau.eval(anchors)
        if np.size(val) == 1:
            val = np.full(next(iter(anchors.values())).shape, val)
        if not np.isfinite(val).all():
            val = np.nan_to_num(val, nan=1.23456789, posinf=2.34567891, neginf=3.45678912)

        if np.isclose(val, val[0], atol=1e-10, rtol=1e-10).all():
            # 常量规则
            if np.isclose(val[0], 1.23456789, atol=1e-10, rtol=1e-10):
                target = Number(np.nan)
            elif np.isclose(val[0], 2.34567891, atol=1e-10, rtol=1e-10):
                target = Number(+np.inf)
            elif np.isclose(val[0], 3.45678912, atol=1e-10, rtol=1e-10):
                target = Number(-np.inf)
            else:
                target = Number(val[0])
            return ReduceRule(tau, target), (None, None, None)
        elif (val_hash := Reduce.get_array_hash(val)) not in hash_dict:
            # 未发生哈希碰撞，追加到 hash_dict 中
            hash_dict[val_hash] = []
            return None, (tau, val, val_hash)
        elif not (candidates := [t for t, v in hash_dict[val_hash] if np.isclose(val, v, atol=1e-8, rtol=1e-5).all()]):
            # 未找到数值相等结果，追加到 hash_dict 中
            return None, (tau, val, val_hash)
        elif len(best_tau := min(candidates, key=lambda x: len(x))) >= len(tau):
            # 数值相等结果无法帮助减少长度，追加到 hash_dict 中
            return None, (tau, val, val_hash)
        else:
            # 存在化简规则 tau -> best_tau
            return ReduceRule(tau, best_tau), (None, None, None)

    @staticmethod
    def process_tau_batch(tau_list: List['Symbol'], anchors: Dict[str, np.ndarray], hash_dict: Dict[str, List[Tuple['Symbol', np.ndarray]]]):
        return [Reduce.process_tau(tau, anchors, hash_dict) for tau in tau_list]
