from .iter_preorder import IterPreorder
from ..symbols import Symbol


class IterPostorder(IterPreorder):
    """
    Post-order iteration over a tree structure.
    Yields each node before its children.
    """
    def generic_visit(self, node: "Symbol", *args, **kwargs):
        for operand in node.operands:
            yield (operand, args, kwargs)
        yield node  # Postorder: yield self last
