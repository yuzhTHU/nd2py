from typing import Dict, List
from ..base_visitor import Visitor, yield_nothing
from ... import Symbol, Number, Variable

class FoldConstant(Visitor):
    """
    访问器，用于将表达式中不含 Number 的子表达式折叠为 Constant。
    """
    def __init__(
        self,
        fold_fitable:bool=True,
        fold_constant:bool=True,
    ):
        self.fold_fitable = fold_fitable
        self.fold_constant = fold_constant

    def __call__(self,
                 node:Symbol,
                 vars:Dict):
        """
        Args:
        - node (Symbol): 要访问的节点。
        - vars (Dict): 变量字典，包含所有变量的值。
        """
        return super().__call__(node, vars)

    def generic_visit(self, node, *args, **kwargs):
        yield from yield_nothing()
        X = []
        for op in node.operands:
            x = yield (op, args, kwargs)
            X.append(x)
        node2 = node.__class__(*X)
        node2.nettype = node.nettype
        if self.fold_constant and  all(isinstance(x, Number) and not x.fitable for x in X):
            y = node2.eval(*args, **kwargs)
            return Number(y, fitable=False, nettype=node2.nettype)
        if self.fold_fitable and all(isinstance(x, Number) and x.fitable for x in X):
            y = node2.eval(*args, **kwargs)
            return Number(y, fitable=True, nettype=node2.nettype)
        return node2
    
    def visit_Empty(self, node:Symbol, *args, **kwargs):
        raise NotImplementedError("Empty node should not be visited.")

    def visit_Number(self, node:Number, *args, **kwargs):
        yield from yield_nothing()
        return node

    def visit_Variable(self, node:Variable, *args, **kwargs):
        yield from yield_nothing()
        if node.name not in kwargs.get('vars', {}):
            return node
        y = node.eval(*args, **kwargs)
        return Number(y, fitable=False, nettype=node.nettype)
