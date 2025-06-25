from ..base_visitor import Visitor, yield_nothing


class GetLength(Visitor):
    def __call__(self, node, *args, **kwargs):
        """Count the number of nodes in the tree."""
        return super().__call__(node, *args, **kwargs)
    
    def generic_visit(self, node, *args, **kwargs):
        yield from yield_nothing()
        children = []
        for child in node.operands:
            child = yield child, args, kwargs
            children.append(child)
        return 1 + sum(children)