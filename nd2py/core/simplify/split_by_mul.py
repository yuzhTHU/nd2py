from typing import List
from functools import reduce
from ..symbols import *
from ..base_visitor import Visitor


class SplitByMul(Visitor):
    def __call__(
        self,
        node: Symbol,
        split_by_div: bool = False,
        merge_coefficients: bool = False,
    ) -> List[Symbol]:
        """Split the node by multiplication, returning a list of symbols.
        Args:
        - node: Symbol, the node to split (a * b * c -> [a, b, c])
        - split_by_div: bool, whether to split by Div (a / b -> [a, b])
        - merge_coefficients: bool, whether to merge coefficients from the symbols
        """
        return super().__call__(
            node,
            split_by_div=split_by_div,
            merge_coefficients=merge_coefficients,
        )

    def generic_visit(self, node: Symbol, *args, **kwargs) -> List[Symbol]:
        return [node]

    def visit_Mul(self, node: Mul, *args, **kwargs) -> List[Symbol]:
        x1, x2 = node.operands
        result = self(x1, *args, **kwargs) + self(x2, *args, **kwargs)
        if kwargs.get("merge_coefficients"):
            result = self.merge_coefficients(result, *args, **kwargs)
        return result

    def visit_Div(self, node: Div, *args, **kwargs) -> List[Symbol]:
        if not kwargs.get("split_by_div"):
            return [node]
        x1, x2 = node.operands
        result1 = self(x1, *args, **kwargs)
        result2 = self(x2, *args, **kwargs)
        for idx, item in enumerate(result2):
            result2[idx] = Inv(item) if not isinstance(item, Inv) else item.operands[0]
        result = result1 + result2
        if kwargs.get("merge_coefficients"):
            result = self.merge_coefficients(result, *args, **kwargs)
        return result

    def merge_coefficients(self, items: List[Symbol], *args, **kwargs) -> List[Symbol]:
        """Merge coefficients from the symbols."""
        is_coeff = [isinstance(item, Number) for item in items]
        if not any(is_coeff):
            return items
        coeff = Number(
            reduce(
                lambda x, y: x * y,
                [item.value for item, flag in zip(items, is_coeff) if flag],
            )
        )
        results = [coeff] + [item for item, flag in zip(items, is_coeff) if not flag]
        return results
