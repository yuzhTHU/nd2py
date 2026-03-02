import random
import logging
import numpy as np
from typing import Tuple
from collections import defaultdict
from ...utils import AttrDict
from ...core.symbols import Add, Sub, Mul, Div, Abs, Inv, Sqrt, Log, Exp, Sin, Arcsin, Cos, Arccos, Tan, Arctan, Pow2, Pow3, Symbol, Empty, Variable, Number
from .metaai_generator import MetaAIGenerator


class SNIPGenerator(MetaAIGenerator):
    def __init__(self, max_var=5, min_unary=0, max_unary=4, 
                 min_binary_per_var=0, max_binary_per_var=1, max_binary_ops_offset=4,
                 max_unary_depth=6, n_mantissa=4, max_exp=1, min_exp=0, **kwargs):
        self.max_var = max_var
        self.min_unary = min_unary
        self.max_unary = max_unary
        self.min_binary_per_var = min_binary_per_var
        self.max_binary_per_var = max_binary_per_var
        self.max_binary_ops_offset = max_binary_ops_offset
        self.max_unary_depth = max_unary_depth
        self.n_mantissa = n_mantissa # 4 -> 0.001 ~ 9.999
        self.max_exp = max_exp # max: 9.999 * 10^1
        self.min_exp = min_exp # min: 0.001 * 10^0
        super().__init__(**kwargs)

    def generate_eqtree(self, n_var=None, n_unary=None, n_binary=None) -> Symbol:
        n_var = n_var or np.random.randint(1, self.max_var)
        n_unary = n_unary or np.random.randint(self.min_unary, self.max_unary+1)
        n_binary = n_binary or np.random.randint(self.min_binary_per_var * n_var, self.max_binary_per_var * n_var + self.max_binary_ops_offset)
        backup, self.unary = self.unary, []
        eqtree = super(SNIPGenerator, self).generate_eqtree(n_binary, n_var)
        self.unary = backup
        eqtree = self.add_unaries(eqtree, n_unary)
        eqtree = self.add_prefactors(eqtree)
        return eqtree

    def generate_float(self) -> Number:
        sign = np.random.choice([-1, 1])
        mantissa = np.random.randint(1, 10 ** self.n_mantissa) / 10 ** (self.n_mantissa-1)
        exponent = np.random.randint(self.min_exp, self.max_exp)
        return Number(sign * mantissa * 10 ** exponent)

    def _add_unaries(self, eqtree:Symbol) -> Symbol:
        for idx, op in enumerate(eqtree.operands):
            if len(op) < self.max_unary_depth:
                unary = np.random.choice(self.unary, p=self.unary_prob)
                eqtree.operands[idx] = unary(self._add_unaries(op))
            else:
                eqtree.operands[idx] = self._add_unaries(op)
        return eqtree

    def add_unaries(self, eqtree:Symbol, n_unary:int) -> Symbol:
        # Add some unary operations
        eqtree = self._add_unaries(eqtree)
        # Remove some unary operations
        postfix = [sym.__class__ if sym.n_operands > 0 else sym for sym in eqtree.postorder()]
        indices = [i for i, x in enumerate(postfix) if x in self.unary]
        if len(indices) > n_unary:
            np.random.shuffle(indices)
            to_remove = indices[:len(indices) - n_unary]
            for index in sorted(to_remove, reverse=True):
                del postfix[index]
        eqtrees = []
        while postfix:
            sym = postfix.pop(0)
            if sym.n_operands == 0:
                eqtrees.append(sym)
            elif sym.n_operands == 1:
                eqtrees[-1] = sym(eqtrees[-1])
            elif sym.n_operands == 2:
                eqtrees[-2] = sym(eqtrees[-2], eqtrees[-1])
                eqtrees.pop(-1)
        assert len(eqtrees) == 1
        return eqtrees[0]
    
    def _add_prefactors(self, eqtree:Symbol) -> Symbol:
        if eqtree.__class__ in [Add, Sub]:
            x1, x2 = eqtree.operands
            if x1.__class__ not in [Add, Sub]:
                eqtree.operands[0] = Mul(self.generate_float(), self._add_prefactors(x1))
            else:
                eqtree.operands[0] = self._add_prefactors(x1)
            eqtree.operands[0].parent = eqtree
            eqtree.operands[0].child_idx = 0
            if x2.__class__ not in [Add, Sub]:
                eqtree.operands[1] = Mul(self.generate_float(), self._add_prefactors(x2))
            else:
                eqtree.operands[1] = self._add_prefactors(x2)
            eqtree.operands[1].parent = eqtree
            eqtree.operands[1].child_idx = 1
            return eqtree
        if eqtree.__class__ in self.unary and eqtree.operands[0].__class__ not in [Add, Sub]:
            a, b = self.generate_float(), self.generate_float()
            eqtree.operands[0] = Add(a, Mul(b, self._add_prefactors(eqtree.operands[0])))
            eqtree.operands[0].parent = eqtree
            eqtree.operands[0].child_idx = 0
            return eqtree
        for idx, op in enumerate(eqtree.operands):
            eqtree.operands[idx] = self._add_prefactors(op)
            eqtree.operands[idx].parent = eqtree
            eqtree.operands[idx].child_idx = idx
        return eqtree

    def add_prefactors(self, eqtree:Symbol) -> Symbol:
        _eqtree = self._add_prefactors(eqtree)
        if list(_eqtree.preorder()) == list(eqtree.preorder()):
            eqtree = Mul(self.generate_float(), eqtree)
        eqtree = Add(self.generate_float(), eqtree)
        return eqtree


class SNIPGenerator2(SNIPGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def add_prefactors(self, eqtree: Symbol) -> Symbol:
        variables = [op for op in eqtree.preorder() if isinstance(op, Variable)]
        for var in variables:
            a = self.generate_float()
            b = self.generate_float()
            var.replace(a * var.copy() + b)
        a = self.generate_float()
        b = self.generate_float()
        return a * eqtree + b


class SNIPGenerator3(SNIPGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_eqtree(self, n_var=None, n_unary=None, n_binary=None) -> Symbol:
        n_var = n_var or np.random.randint(1, self.max_var)
        n_unary = n_unary or np.random.randint(self.min_unary, self.max_unary+1)
        n_binary = n_binary or np.random.randint(self.min_binary_per_var * n_var, self.max_binary_per_var * n_var + self.max_binary_ops_offset)
        backup, self.unary = self.unary, []
        eqtree = super(SNIPGenerator, self).generate_eqtree(n_binary, n_var)
        self.unary = backup
        eqtree = self.add_unaries(eqtree, n_unary)
        # eqtree = self.add_prefactors(eqtree)
        return eqtree

    def generate_float(self) -> Number:
        raise NotImplementedError

