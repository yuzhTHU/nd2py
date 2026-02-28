import json
import time
import random
import logging
import sklearn
import traceback
import numpy as np
import sympy as sp
import nd2py as nd
import pandas as pd
from tqdm import tqdm
from typing import List, Generator, Tuple, Dict, Optional, Literal
from nd2py.utils import seed_all, Timer, NamedTimer, R2_score, ParallelTimer
from ...core.symbols import *

__all__ = ["MCTS"]


def simplify(eq: nd.Symbol):
    try:
        expr = sp.parse_expr(eq.to_str())
        expr = sp.simplify(expr)
        return nd.parse_expr(str(expr))
    except:
        return eq.copy()


class Node:
    def __init__(self, eqtree: nd.Symbol):
        # Formula part
        self.eqtree = eqtree
        self.fitted_eqtree = None
        self.complexity = None
        self.reward = None
        self.r2 = None

        # MC Tree part
        self.parent = None
        self.children = []
        self.N = 0
        self.Q = 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.eqtree) + f" (N={self.N}, Q={self.Q/(self.N+1e-6):.2f})"

    def UCT(self, c) -> float:
        if self.parent is None:
            return float("inf")
        return self.Q / (self.N + 1e-6) + c * np.sqrt(
            np.log(self.parent.N) / (self.N + 1e-6)
        )

    def to_route(self, N=5, c=1.41) -> str:
        """
        Root
        ├ Node1
        ┆ ├ self
        ┆ └ Node1-2
        └ Node2
        """
        rev_route = [self]
        tmp = self
        while tmp.parent:
            rev_route.append(tmp.parent)
            tmp = tmp.parent
        items = []
        for node in rev_route:
            if node.parent:
                siblings = node.parent.children
                UCT = {x: x.UCT(c) for x in siblings}
                siblings = sorted(siblings, key=UCT.get, reverse=True)
                siblings = siblings[:N]
            else:
                siblings = [node]
                UCT = {node: 0.0}
            new_items = [f"{node} (UCT={UCT[node]:.2f})" for node in siblings]
            self_idx = siblings.index(node)
            for idx, item in enumerate(items):
                items[idx] = ("├ " if idx < len(items) - 1 else "└ ") + item.replace(
                    "\n", "\n" + ("┆ " if idx < len(items) - 1 else "  ")
                )
            new_items[self_idx] = (
                "\033[31m"
                + new_items[self_idx]
                + "\033[0m"
                + ("\n" if items else "")
                + "\n".join(items)
            )
            items = new_items
        assert len(items) == 1
        return items[0]

    def copy(self) -> "Node":
        copy = Node(self.eqtree.copy())
        copy.complexity = self.complexity
        copy.reward = self.reward
        copy.r2 = self.r2
        return copy


class MCTS(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """
    Monte Carlo Tree Search-based Symbolic Regression
    """

    def __init__(
        self,
        variables: List[Variable],
        binary: List[Symbol] = [Add, Sub, Mul, Div, Max, Min],
        unary: List[Symbol] = [Sqrt, Log, Abs, Neg, Inv, Sin, Cos, Tan],
        max_params: int = 2,
        const_range: Tuple[float, float] = (-1.0, 1.0),
        depth_range: Tuple[int, int] = (2, 6),
        nettype: Optional[Literal["node", "edge", "scalar"]] = "scalar",
        log_per_iter: int = float("inf"),
        log_per_sec: float = float("inf"),
        log_detailed_speed: bool = False,
        save_path: str = None,
        random_state: Optional[int] = None,
        n_iter=100,
        use_tqdm=False,
        edge_list: Tuple[List[int], List[int]] = None,
        num_nodes: int = None,
        time_limit=None,
        sample_num=300,
        keep_vars=False,
        normalize_y=False,
        normalize_X=False,
        remove_abnormal=False,
        train_eval_split=1.0,

        child_num=50,
        n_playout=100,
        d_playout=10,
        max_len=30,
        c=1.41,
        eta=0.999,
        **kwargs,
    ):
        self.eqtree = None
        self.variables = variables
        self.binary = binary
        self.unary = unary
        self.max_params = max_params
        self.const_range = const_range
        self.depth_range = depth_range
        self.nettype = nettype
        self.log_per_iter = log_per_iter
        self.log_per_sec = log_per_sec
        self.log_detailed_speed = log_detailed_speed
        self.save_path = save_path
        self.random_state = random_state
        self.n_iter = n_iter
        self.use_tqdm = use_tqdm
        self.edge_list = edge_list
        self.num_nodes = num_nodes
        self.time_limit = time_limit
        self.sample_num = sample_num
        self.keep_vars = keep_vars
        self.normalize_y = normalize_y
        self.normalize_X = normalize_X
        self.remove_abnormal = remove_abnormal
        self.train_eval_split = train_eval_split

        self.child_num = child_num
        self.n_playout = n_playout
        self.d_playout = d_playout
        self.max_len = max_len
        self.c = c
        self.eta = eta
        if kwargs:
            self.logger.warning(
                "Unknown args: %s", ", ".join(f"{k}={v}" for k, v in kwargs.items())
            )

        self.records = []
        self.logger = logging.getLogger(__name__)
        self.step_timer = Timer()
        self.view_timer = Timer()
        self.named_timer = NamedTimer()
        self.para_timer = ParallelTimer()

    def __repr__(self):
        res = "None" if self.eqtree is None else self.eqtree.to_str()
        return "{}({})".format(self.__class__.__name__, res)

    def fit(
        self,
        X: np.ndarray | pd.DataFrame | Dict[str, np.ndarray],
        y: np.ndarray | pd.Series,
    ):
        """
        Args:
            X: (n_samples, n_dims)
            y: (n_samples,)
        """
        if isinstance(X, np.ndarray):
            X = {var.name: x for var, x in zip(self.variables, X[..., :])}
        elif isinstance(X, pd.DataFrame):
            X = {col: X[col].values for i, col in enumerate(X.columns)}
        elif isinstance(X, dict):
            X = {k: np.asarray(v) for k, v in X.items()}
        else:
            raise ValueError(f"Unknown type: {type(X)}")

        # Root Node
        self.MC_tree = Node(nd.Empty(nettype=self.nettype))

        # Search
        stop = False
        best_node = None
        self.eqtree = None
        self.named_timer.clear(reset_last_add_time=True)
        # self.speed_timer.clear(reset_last_add_time=True)
        self.start_time = time.time()
        for iter in tqdm(range(1, self.n_iter + 1), disable=not self.use_tqdm):
            leaf = self.select(self.MC_tree)
            expand = self.expand(leaf, X, y)
            reward, best_simulated = self.simulate(expand, X, y)
            self.backpropagate(expand, reward)
            self.para_timer.add('iteration')

            ## Prepare log & record
            record = dict( # {"iter": iter, "time": time.time() - self.start_time}
                iter=iter,
                time=self.para_timer.time,
                speed=self.para_timer.named_speed,
                time_usage=self.para_timer.named_time,
                call_count=self.para_timer.named_count,
            )

            log = {"Iter": iter}

            if _update_best := (best_node is None or best_simulated.reward > best_node.reward):
                best_node = best_simulated # .copy()
                self.eqtree = best_node.fitted_eqtree

                # self.set_reward(self.eqtree, X, y)
                # self.eqtree = simplify(self.eqtree.fitted_eqtree)
                record["eqtree"] = str(best_node.eqtree)
                record["fitted_eqtree"] = str(best_node.fitted_eqtree)
                record["complexity"] = best_node.complexity
                record["reward"] = best_node.reward
                record["r2"] = best_node.r2

                log['Eqtree'] = str(best_node.eqtree)
                log['Fitted eqtree'] = str(best_node.fitted_eqtree)
                log['Complexity'] = best_node.complexity
                log['Reward'] = f"{best_node.reward:.5f}"
                log['R2'] = f"{best_node.r2:.5f}"

            _early_stop = (
                (iter == self.n_iter)
                or self.time_limit and (self.para_timer.time > self.time_limit)
            )
                                    
            if (
                _update_best or _early_stop
                or self.log_per_iter and (not iter % self.log_per_iter)
                or self.log_per_sec and ('_last_log_time' not in locals() or time.time() - _last_log_time > self.log_per_sec)
            ):
                log["Speed"] = self.para_timer.to_str('time', 'speed', None)
                log['Time Usage'] = self.named_timer.to_str('pace', 'time', 'by_time')
                msg = " | ".join(f"\033[4m{k}\033[0m={v}" for k, v in log.items())
                (self.logger.note if _update_best else self.logger.info)(msg)

            self.records.append(record)
            if self.save_path:
                with open(self.save_path / 'records.jsonl', "a") as f:
                    f.write(json.dumps(record) + "\n")

            if _early_stop:
                self.logger.note(f"Early stop at iter {iter}")
                break

    def predict(
        self, X: np.ndarray | pd.DataFrame | Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Args:
            X: (n_samples, n_dims)
        Returns:
            y: (n_samples,)
        """
        if self.eqtree is None:
            raise ValueError("Model not fitted yet")
        if isinstance(X, np.ndarray):
            X = {var.name: x for var, x in zip(self.variables, X[..., :])}
        elif isinstance(X, pd.DataFrame):
            X = {col: X[col].values for i, col in enumerate(X.columns)}
        elif isinstance(X, dict):
            X = {k: np.asarray(v) for k, v in X.items()}
        else:
            raise ValueError(f"Unknown type: {type(X)}")
        return self.eqtree.eval(
            vars=X, edge_list=self.edge_list, num_nodes=self.num_nodes
        )

    def action(self, state: Node, action: Tuple[nd.Symbol, nd.Symbol]) -> Node:
        """
        将 action[0]:nd.Empty 替换为 action[1]:nd.Symbol
        """
        state2 = state.copy()
        e, op = action
        if e is None and op is None:
            return state2
        e2 = state2.eqtree.get_path(state.eqtree.path_to(e))
        state2.eqtree = state2.eqtree.replace(e2, op)
        return state2

    def check_valid_action(
        self, state: Node, action: Tuple[nd.Symbol, nd.Symbol]
    ) -> bool:
        e, op = action
        # empty_list = [i for i in state.eqtree.iter_preorder() if isinstance(i, nd.Empty)]
        if len(state.eqtree) + op.n_operands > self.max_len:
            return False
        if not e.replaceable_nettype() & op.nettype_range:
            return False
        # if (e is None and op is None) ^ (len(empty_list) == 0): return False
        return True

    def iter_valid_action(
        self, state: Node, shuffle=False
    ) -> Generator[Tuple[nd.Symbol, nd.Symbol], None, None]:
        empty_list = [
            i for i in state.eqtree.iter_preorder() if isinstance(i, nd.Empty)
        ]
        # if len(empty_list) == 0:
        #     yield None, None
        #     return

        loader = []
        for e in empty_list:
            for op in self.binary + self.unary + self.variables:
                if e.nettype not in op.nettype_range:
                    continue
                if isinstance(op, type):
                    op = op(nettype=e.nettype)
                    for operand in op.operands:
                        replaceable_nettype = op.replaceable_nettype(operand)
                        operand.nettype = random.choice(list(replaceable_nettype))
                loader.append((e, op))
        if shuffle:
            random.shuffle(loader)
        for e, op in loader:
            if self.check_valid_action(state, (e, op)):
                yield e, op

    def pick_valid_action(self, state: Node) -> Tuple[nd.Symbol, nd.Symbol]:
        empty_list = [
            i for i in state.eqtree.iter_preorder() if isinstance(i, nd.Empty)
        ]
        # if len(empty_list) == 0: return None, None
        op_list = self.binary + self.unary + self.variables
        for _ in range(1000):
            e = random.choice(empty_list)
            op = random.choice(op_list)
            if e.nettype not in op.nettype_range:
                continue
            if isinstance(op, type):
                op = op(nettype=e.nettype)
                for operand in op.operands:
                    replaceable_nettype = op.replaceable_nettype(operand)
                    operand.nettype = random.choice(list(replaceable_nettype))
            if self.check_valid_action(state, (e, op)):
                break
        else:
            raise ValueError("Cannot find valid action")
        return e, op

    def fill_to_complete(self, state: Node) -> Node:
        state2 = state.copy()
        empty_list = [
            i for i in state2.eqtree.iter_preorder() if isinstance(i, nd.Empty)
        ]
        for e in empty_list:
            operands = [var for var in self.variables if var.nettype in e.replaceable_nettype()]
            if not operands:
                continue
            op = random.choice(operands)
            state2.eqtree = state2.eqtree.replace(e, op)
        return state2

    def select(self, root: Node) -> Node:
        node = root
        while node.children:
            node = max(node.children, key=lambda x: x.UCT(self.c))
        return node

    def expand(self, node: Node, X: Dict[str, np.ndarray], y: np.ndarray) -> Node:
        for idx, action in enumerate(self.iter_valid_action(node, shuffle=True)):
            child = self.action(node, action)
            child.parent = node
            child.xchild = len(node.children)
            node.children.append(child)
            if self.child_num and idx + 1 >= self.child_num:
                break
        if not node.children:
            return node  # leaf node
        return random.choice(node.children)

    def simulate(
        self, node: Node, X: Dict[str, np.ndarray], y: np.ndarray
    ) -> Tuple[Node, float]:
        if getattr(node, "reward", None) is not None:
            return node.reward, node
        best = None
        for i in range(self.n_playout):
            state = node
            for j in range(self.d_playout):
                if not any(
                    isinstance(i, nd.Empty) for i in state.eqtree.iter_preorder()
                ):
                    break
                action = self.pick_valid_action(state)
                state = self.action(state, action)
            else:
                state = self.fill_to_complete(state)
            self.set_reward(state, X, y)
            if best is None or state.reward > best.reward:
                best = state
        if best is None:
            best = Node(nd.Number(0))
            self.set_reward(best, X, y)
            self.logger.warning(
                f"best is None after simualte({node}), set best to {best}"
            )
        return best.reward, best

    def backpropagate(self, node: Node, reward: float):
        while node:
            node.N += 1
            node.Q += reward
            node = node.parent

    def set_reward(self, node: Node, X: Dict[str, np.ndarray], y: np.ndarray):
        self.view_timer.add(1)

        if any(isinstance(i, nd.Empty) for i in node.eqtree.iter_preorder()):
            node.r2 = -float('inf')
            node.reward = -float('inf')
            node.complexity = float("inf")
            return

        bfgs = nd.BFGSFit(node.eqtree, options={"maxiter": 100}, fold_constant=True, edge_list=self.edge_list, num_nodes=self.num_nodes)
        bfgs.fit(X, y)
        y_pred = bfgs.expression.eval(X, edge_list=self.edge_list, num_nodes=self.num_nodes)
        y_true = y
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            node.r2 = R2_score(y_true, y_pred).clip(0, 1)
            node.fitted_eqtree = bfgs.expression
            node.complexity = len(node.fitted_eqtree)
            node.reward = self.eta**node.complexity / (2 - node.r2)
        if not np.isfinite(node.reward):
            node.reward = 0.0
