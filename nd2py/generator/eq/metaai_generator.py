import random
import logging
import numpy as np
from typing import Tuple, List, Generator, Dict
from collections import defaultdict
from ...utils import AttrDict
from ...core.symbols import Add, Sub, Mul, Div, Abs, Inv, Sqrt, Log, Exp, Sin, Arcsin, Cos, Arccos, Tan, Arctan, Pow2, Pow3, Symbol, Empty, Variable, Identity


def decompose(eqtrees:Symbol|List[Symbol], 
              route:List[List[Symbol]]=[],
              route2:List[Symbol]=[Empty()]) -> Generator[List[Tuple[List[Symbol], Symbol]], None, None]:
    """
    将 eqtrees 分解为 Leaf Symbol 的组合，尝试所有可能的分解方式，每次返回一种分解方式
    Input: exp(x+y)*(x+z)
    Output1:
        [exp(x + y) * (x + z)] | ?
        [exp(x + y), x + z] | ? * ?
        [x + y, x + z] | exp(?) * ?
        [x, y, x + z] | exp(? + ?) * ?
        [x, y, x, z] | exp(? + ?) * (? + ?)
    Output2:
        [exp(x + y) * (x + z)] | ?
        [exp(x + y), x + z] | ? * ?
        [x + y, x + z] | exp(?) * ?
        [x + y, x, z] | exp(?) * (? + ?)
        [x, y, x, z] | exp(? + ?) * (? + ?)
    Output3:
        [exp(x + y) * (x + z)] | ?
        [exp(x + y), x + z] | ? * ?
        [exp(x + y), x, z] | ? * (? + ?)
        [x + y, x, z] | exp(?) * (? + ?)
        [x, y, x, z] | exp(? + ?) * (? + ?)
    """
    if isinstance(eqtrees, Symbol): eqtrees = [eqtrees]
    if all([eq.n_operands == 0 for eq in eqtrees]): 
        yield list(zip([*route, eqtrees], route2))
    else:
        for idx, eq in enumerate(eqtrees):
            if eq.n_operands == 0: continue
            f = route2[-1].copy()
            cnt = -1
            for n in f.preorder():
                if isinstance(n, Empty): 
                    cnt +=1
                    if cnt == idx: 
                        if n.parent: n.replace(eq.__class__())
                        else: f = eq.__class__()
                        break
            else:
                raise Exception('Error')
            yield from decompose([*eqtrees[:idx], *eq.operands, *eqtrees[idx+1:]], 
                                 [*route, eqtrees],
                                 [*route2, f])



class MetaAIGenerator:
    def __init__(self, operators_to_downsample='Div:0,Arcsin:0,Arccos:0,Tan:0.2,Arctan:0.2,Sqrt:5,Pow2:3,Inv:3', **kwargs):
        self.binary = [Add, Sub, Mul, Div]
        self.unary = [Abs, Inv, Sqrt, Log, Exp, Sin, Arcsin, Cos, Arccos, Tan, Arctan, Pow2, Pow3]
        
        prob_dict = defaultdict(lambda: 1.0)
        for item in operators_to_downsample.split(","):
            if item != "":
                op, prob = item.split(':')
                prob_dict[eval(op)] = float(prob)
        self.binary_prob = [prob_dict[op] for op in self.binary]
        self.binary_prob = np.array(self.binary_prob) / sum(self.binary_prob)
        self.unary_prob = [prob_dict[op] for op in self.unary]
        self.unary_prob = np.array(self.unary_prob) / sum(self.unary_prob)


    def generate_eqtree(self, n_operators, n_var) -> Symbol:
        sentinel = Identity(); # 哨兵节点

        # construct unary-binary tree
        empty_nodes = [*sentinel.operands]
        next_en = -1
        n_empty = 1
        while n_operators > 0:
            next_pos, arity = self.generate_next_pos(n_empty, n_operators)
            op = self.generate_ops(arity)
            next_en += next_pos + 1
            n_empty -= next_pos + 1
            empty_nodes[next_en] = empty_nodes[next_en].replace(op())
            empty_nodes.extend(empty_nodes[next_en].operands)
            n_empty += op.n_operands
            n_operators -= 1
        
        # fill variables
        n_used_var = 0
        for n in empty_nodes:
            if isinstance(n, Empty):
                sym, n_used_var = self.generate_leaf(n_var, n_used_var)
                n.replace(sym)

        return sentinel.operands[0]

    def dist(self, n_op, n_emp):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[n][e] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(n, 0) = 0
            D(0, e) = 1
            D(n, e) = D(n, e - 1) + p_1 * D(n - 1, e) + D(n - 1, e + 1)
        p1 = 0 if binary trees, or 1 if unary-binary trees
        """
        if not hasattr(self, 'dp_cache'): self.dp_cache = [[0]]
        p1 = 1 if self.unary else 0
        if len(self.dp_cache) <= n_op + n_emp:
            for _ in range(len(self.dp_cache), n_op + n_emp + 1):
                self.dp_cache[0].append(1)
                for r, row in enumerate(self.dp_cache[1:], 1):
                    row.append(row[-1] + p1 * self.dp_cache[r-1][-2] + self.dp_cache[r-1][-1])
                self.dp_cache.append([0])
        return self.dp_cache[n_op][n_emp]

    def generate_leaf(self, n_var:int, n_used_var:int) -> Tuple[Symbol, int]:
        if n_used_var < n_var:
            return Variable(f"x_{n_used_var+1}"), n_used_var+1
        else:
            idx = np.random.randint(1, n_var + 1)
            return Variable(f"x_{idx}"), n_used_var

    def generate_ops(self, n_operands:int) -> Symbol:
        if n_operands == 1:
            return np.random.choice(self.unary, p=self.unary_prob)
        else:
            return np.random.choice(self.binary, p=self.binary_prob)

    def generate_next_pos(self, n_empty, n_operators):
        """
        Sample the position of the next node (binary case).
        Sample a position in {0, ..., `n_empty` - 1}.
        """
        assert n_empty > 0
        assert n_operators > 0
        probs = [self.dist(n_operators - 1, n_empty - i + 1) for i in range(n_empty)]
        if self.unary:
            probs += [self.dist(n_operators - 1, n_empty - i) for i in range(n_empty)]
        probs = np.array(probs, dtype=np.float64) / self.dist(n_operators, n_empty)
        next_pos = np.random.choice(len(probs), p=probs)
        n_operands = 1 if next_pos >= n_empty else 2
        next_pos %= n_empty
        return next_pos, n_operands

