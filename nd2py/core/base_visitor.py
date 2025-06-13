from abc import ABC, abstractmethod
from .symbols import Symbol


class Visitor(ABC):
    def __call__(self, node: Symbol, *args, **kwargs):
        """
        1) call Visitor.visit_<ClassName> based on type(node)
        2) call Visitor.generic_visit if no Visitor.visit_<ClassName> defined
        """
        method = getattr(self, "visit_" + type(node).__name__, self.generic_visit)
        return method(node, *args, **kwargs)

    @abstractmethod
    def generic_visit(self, node: Symbol, *args, **kwargs):
        msg = f"generic_visit not implemented for {type(self).__name__}"
        raise NotImplementedError(msg)
