from typing import Dict, List
from ..base_visitor import Visitor
from ... import Symbol, Number, Variable

class FoldConstant(Visitor):
    """
    访问器，用于将表达式中不含 Number 的子表达式折叠为 Constant。
    """
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
        X = [self(op, *args, **kwargs) for op in node.operands]
        node2 = node.__class__(*X)
        node2.nettype = node.nettype
        if all(isinstance(x, Number) and not x.fitable for x in X):
            y = node2.eval(*args, **kwargs)
            return Number(y, fitable=False, nettype=node2.nettype)
        return node2
    
    def visit_Empty(self, node:Symbol, *args, **kwargs):
        raise NotImplementedError("Empty node should not be visited.")

    def visit_Number(self, node:Number, *args, **kwargs):
        return node

    def visit_Variable(self, node:Variable, *args, **kwargs):
        y = node.eval(*args, **kwargs)
        return Number(y, fitable=False, nettype=node.nettype)
