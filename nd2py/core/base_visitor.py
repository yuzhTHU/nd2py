from abc import ABC, abstractmethod
from .symbols import Symbol


def yield_nothing():
    """A generator that yields nothing, used as a placeholder for methods that do not yield."""
    if False:
        yield


class Visitor(ABC):
    def __call__(self, node: Symbol, *args, **kwargs):
        """
        1) call Visitor.visit_<ClassName> based on type(node)
        2) call Visitor.generic_visit if no Visitor.visit_<ClassName> defined
        """
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
                        f"Expected a generator but got {type(gen).__name__}, please add `yield from yield_nothing()` in {type(self).__name__}.{method.__name__}."
                    )
                try:
                    child, args, kwargs = next(gen)
                    stack.append(("resume", node, None, None, gen))
                    stack.append(("start", child, args, kwargs, None))
                except StopIteration as e:
                    result = e.value
            elif state == "resume":
                try:
                    child, args, kwargs = gen.send(result)
                    stack.append(("resume", node, None, None, gen))
                    stack.append(("start", child, args, kwargs, None))
                except StopIteration as e:
                    result = e.value
            else:
                raise ValueError(f"Unknown state: {state}")
        return result

    @abstractmethod
    def generic_visit(self, node: Symbol, *args, **kwargs):
        msg = f"generic_visit not implemented for {type(self).__name__}"
        raise NotImplementedError(msg)
