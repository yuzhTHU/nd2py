from ..base_visitor import Visitor, yield_nothing
from ..symbols import Symbol


class IterPreorder(Visitor):
    """
    Pre-order iteration over a tree structure.
    Yields each node before its children.
    """

    def __call__(self, node: Symbol, *args, **kwargs):
        """Non-recursive preorder traversal of the Symbol tree using an explicit stack."""
        stack = [("start", node, args, kwargs, None)]
        result = None
        while stack:
            state, node, args, kwargs, gen = stack.pop()
            if state == "start":
                method = getattr(
                    self, "visit_" + type(node).__name__, self.generic_visit
                )
                gen = method(node, *args, **kwargs)
                if not hasattr(gen, "__next__"):
                    raise TypeError(
                        f"Expected a generator but got {type(gen).__name__}, please add `yield from yield_nothing()` in {method.__name__}."
                    )
                try:
                    yielded = next(gen)
                    if self.is_yielded_node(yielded):
                        # Yield this node to caller
                        yield yielded
                        stack.append(("resume", node, None, None, gen))
                    else:
                        # Recurse into yielded child node
                        child, args, kwargs = yielded
                        stack.append(("resume", node, None, None, gen))
                        stack.append(("start", child, args, kwargs, None))
                except StopIteration:
                    pass
            elif state == "resume":
                try:
                    yielded = gen.send(result)
                    if self.is_yielded_node(yielded):
                        yield yielded
                        stack.append(("resume", node, None, None, gen))
                    else:
                        child, args, kwargs = yielded
                        stack.append(("resume", node, None, None, gen))
                        stack.append(("start", child, args, kwargs, None))
                except StopIteration:
                    pass
            else:
                raise ValueError(f"Unknown state: {state}")

    def is_yielded_node(self, yielded):
        # If yielded is not a tuple, treat it as visit result
        return not (isinstance(yielded, tuple) and len(yielded) == 3)

    def generic_visit(self, node: "Symbol", *args, **kwargs):
        yield node  # Preorder: yield self first
        for operand in node.operands:
            yield (operand, args, kwargs)
