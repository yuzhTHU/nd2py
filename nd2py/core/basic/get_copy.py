from copy import deepcopy
from ..base_visitor import Visitor, yield_nothing


class GetCopy(Visitor):
    def __call__(self, node, *args, **kwargs):
        """Create a copy of the node."""
        return super().__call__(node, *args, **kwargs)

    def generic_visit(self, node, *args, **kwargs):
        yield from yield_nothing()
        children = []
        for child in node.operands:
            child = yield child, args, kwargs
            children.append(child)
        return node.__class__(*children)

    def visit_Number(self, node, *args, **kwargs):
        yield from yield_nothing()
        return node.__class__(
            deepcopy(node.value), nettype=node.nettype, fitable=node.fitable
        )

    def visit_Variable(self, node, *args, **kwargs):
        yield from yield_nothing()
        return node.__class__(node.name, nettype=node.nettype)

    def visit_Empty(self, node, *args, **kwargs):
        yield from yield_nothing()
        return node.__class__(nettype=node.nettype)
