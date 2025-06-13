from typing import List
from ..symbols import *
from ..base_visitor import Visitor


class SplitByAdd(Visitor):
    def __call__(
        self,
        node: Symbol,
        split_by_sub: bool = False,
        expand_mul: bool = False,
        expand_div: bool = False,
        expand_aggr: bool = False,
        expand_rgga: bool = False,
        expand_sour: bool = False,
        expand_targ: bool = False,
        expand_readout: bool = False,
        remove_coefficients: bool = False,
        merge_bias: bool = False,
    ) -> List[Symbol]:
        """Split the node by addition, returning a list of symbols.
        Args:
        - node: Symbol, the node to split (a + b + c -> [a, b, c])
        - split_by_sub: bool, whether to split by Sub (a - b -> [a, b])
        - expand_mul: bool, whether to expand Mul ((a+b) * (c + d) -> [a*c, a*d, b*c, b*d])
        - expand_div: bool, whether to expand Div ((a+b) / (c + d) -> [a/(c+d), b/(c+d)])
        - expand_aggr: bool, whether to expand Aggr (Aggr(a + b) -> [Aggr(a), Aggr(b)])
        - expand_rgga: bool, whether to expand Rgga (Rgga(a + b) -> [Rgga(a), Rgga(b)])
        - expand_sour: bool, whether to expand Sour (Sour(a + b) -> [Sour(a), Sour(b)])
        - expand_targ: bool, whether to expand Targ (Targ(a + b) -> [Targ(a), Targ(b)])
        - expand_readout: bool, whether to expand Readout (Readout(a + b) -> [Readout(a), Readout(b)])
        - remove_coefficients: bool, whether to remove coefficients from the symbols
        - merge_bias: bool, whether to merge bias terms
        """
        return super().__call__(
            node,
            split_by_sub=split_by_sub,
            expand_mul=expand_mul,
            expand_div=expand_div,
            expand_aggr=expand_aggr,
            expand_rgga=expand_rgga,
            expand_sour=expand_sour,
            expand_targ=expand_targ,
            expand_readout=expand_readout,
            remove_coefficients=remove_coefficients,
            merge_bias=merge_bias,
        )

    def generic_visit(self, node: Symbol, *args, **kwargs) -> List[Symbol]:
        return [node]

    def visit_Add(self, node: Add, *args, **kwargs) -> List[Symbol]:
        x1, x2 = node.operands
        result = self(x1, *args, **kwargs) + self(x2, *args, **kwargs)
        if kwargs.get("merge_bias"):
            result = self.merge_bias(result, *args, **kwargs)
        return result

    def visit_Sub(self, node: Sub, *args, **kwargs) -> List[Symbol]:
        if not kwargs.get("split_by_sub"):
            return [self]
        x1, x2 = node.operands
        result1 = self(x1, *args, **kwargs)
        result2 = self(x2, *args, **kwargs)
        for idx, item in enumerate(result2):
            result2[idx] = Neg(item) if not isinstance(item, Neg) else item.operands[0]
        result = result1 + result2
        if kwargs.get("merge_bias"):
            result = self.merge_bias(result, *args, **kwargs)
        return result

    def visit_Mul(self, node: Mul, *args, **kwargs) -> List[Symbol]:
        if not kwargs.get("expand_mul"):
            return [node]
        x1, x2 = node.operands
        result1 = self(x1, *args, **kwargs)
        result2 = self(x2, *args, **kwargs)
        result = []
        for item in result1:
            for jtem in result2:
                if not kwargs.get("remove_coefficients"):
                    result.append(item * jtem)
                elif isinstance(item, Number):
                    result.append(jtem)
                elif isinstance(jtem, Number):
                    result.append(item)
                else:
                    result.append(item * jtem)
        if kwargs.get("merge_bias"):
            result = self.merge_bias(result, *args, **kwargs)
        return result

    def visit_Div(self, node: Div, *args, **kwargs) -> List[Symbol]:
        if not kwargs.get("expand_div"):
            return [node]
        x1, x2 = node.operands
        result = self(x1, *args, **kwargs)
        for idx, item in result:
            if not kwargs.get("remove_coefficients"):
                result[idx] = item / x2
            elif isinstance(item, Number):
                result[idx] = Inv(x2)
            elif isinstance(x2, Number):
                result[idx] = item
            else:
                result[idx] = item / x2
        if kwargs.get("merge_bias"):
            result = self.merge_bias(result, *args, **kwargs)
        return result

    def visit_Sour(self, node: Sour, *args, **kwargs) -> List[Symbol]:
        if not kwargs.get("expand_sour"):
            return [node]
        result = self(node.operands[0], *args, **kwargs)
        if kwargs.get("merge_bias"):
            result = self.merge_bias(result, *args, **kwargs)
        for idx, item in enumerate(result):
            if item.nettype != "scalar":
                result[idx] = Sour(item)
        return result

    def visit_Targ(self, node: Targ, *args, **kwargs) -> List[Symbol]:
        if not kwargs.get("expand_targ"):
            return [node]
        result = self(node.operands[0], *args, **kwargs)
        if kwargs.get("merge_bias"):
            result = self.merge_bias(result, *args, **kwargs)
        for idx, item in enumerate(result):
            if item.nettype != "scalar":
                result[idx] = Targ(item)
        return result

    def visit_Aggr(self, node: Aggr, *args, **kwargs) -> List[Symbol]:
        if not kwargs.get("expand_aggr"):
            return [node]
        result = self(node.operands[0], *args, **kwargs)
        if kwargs.get("merge_bias"):
            result = self.merge_bias(result, *args, **kwargs)
        for idx, item in enumerate(result):
            result[idx] = Aggr(item)  # Aggr(C) 和 C 数学不等价
        return result

    def visit_Rgga(self, node: Rgga, *args, **kwargs) -> List[Symbol]:
        if not kwargs.get("expand_rgga"):
            return [node]
        result = self(node.operands[0], *args, **kwargs)
        if kwargs.get("merge_bias"):
            result = self.merge_bias(result, *args, **kwargs)
        for idx, item in enumerate(result):
            result[idx] = Rgga(item)  # Rgga(C) 和 C 数学不等价
        return result

    def visit_Readout(self, node: Readout, *args, **kwargs) -> List[Symbol]:
        if not kwargs.get("expand_readout"):
            return [node]
        result = self(node.operands[0], *args, **kwargs)
        if kwargs.get("merge_bias"):
            result = self.merge_bias(result, *args, **kwargs)
        for idx, item in enumerate(result):
            result[idx] = Readout(item)
            if item.nettype == "scalar" and not kwargs.get("remove_coefficients"):
                # Readout(x + 1) = Readout(x) + Readout(1) * x.shape[-1], differ by a constant factor
                raise NotImplementedError(
                    "Readout(scalar) and Readout(node/edge) will differ by a constant factor, "
                    "you can set remove_coefficients=True to ignore this."
                )
        return result

    def merge_bias(self, items: List[Symbol], *args, **kwargs) -> List[Symbol]:
        """Merge bias terms in the node."""
        is_bias = [isinstance(item, Number) for item in items]
        if not any(is_bias):
            return items
        bias = Number(sum(item.value for item, flag in zip(items, is_bias) if flag))
        if kwargs.get("remove_coefficients"):
            bias = Number(1.0)
        results = [bias] + [item for item, flag in zip(items, is_bias) if not flag]
        return results
