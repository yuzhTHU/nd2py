# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
from typing import List, Generator, Tuple, Dict, Type, TYPE_CHECKING
from ..base_visitor import Visitor, yield_nothing
from functools import partialmethod, reduce

if TYPE_CHECKING:
    from ..symbols import *
    _YieldType = Tuple[Symbol, Tuple, Dict]  # (node, args, kwargs)
    _SendType = List[Symbol]  # List of symbols
    _ReturnType = List[Symbol]  # Merged list of symbols
    _Type = Generator[_YieldType, _SendType, _ReturnType]


class Simplify(Visitor):
    def __call__(
        self,
        node: Symbol,
        transform_constant_subtree: bool = True,
        remove_useless_readout: bool = True,
        remove_nested_sin: bool = False,
        remove_nested_cos: bool = False,
        remove_nested_tanh: bool = False,
        remove_nested_sigmoid: bool = False,
        remove_nested_sqrt: bool = False,
        remove_nested_sqrtabs: bool = False,
        remove_nested_exp: bool = False,
        remove_nested_log: bool = False,
        remove_nested_logabs: bool = False,
    ) -> List[Symbol]:
        return super().__call__(
            node,
            transform_constant_subtree=transform_constant_subtree,
            remove_nested_sin=remove_nested_sin,
            remove_nested_cos=remove_nested_cos,
            remove_nested_tanh=remove_nested_tanh,
            remove_nested_sigmoid=remove_nested_sigmoid,
            remove_nested_sqrt=remove_nested_sqrt,
            remove_nested_sqrtabs=remove_nested_sqrtabs,
            remove_nested_exp=remove_nested_exp,
            remove_nested_log=remove_nested_log,
            remove_nested_logabs=remove_nested_logabs,
            remove_useless_readout=remove_useless_readout,
        )

    def generic_visit(self, node: Symbol, *args, **kwargs) -> _Type:
        yield from yield_nothing()
        results = []
        for x in node.operands:
            result = yield (x, args, kwargs)
            results.append(result)

        if kwargs["transform_constant_subtree"] and all(
            type(result).__name__ == 'Number' for result in results
        ):
            Number = self._get_symbol('Number')
            value = sum(result.value for result in results)
            return Number(value)
        return node.__class__(*results)

    def remove_nested_unary(
        self, node: Symbol, *args, **kwargs
    ) -> _Type:
        unary = type(node)
        x = node.operands[0]
        result = yield (x, args, kwargs)

        if kwargs["transform_constant_subtree"] and type(result).__name__ == 'Number':
            Number = self._get_symbol('Number')
            return Number(unary(result).eval())

        if kwargs[f"remove_nested_{unary.__name__.lower()}"]:
            loader = result.iter_preorder()
            if any(isinstance(op, unary) for op in loader):
                return result
        return unary(result)

    visit_Sin = remove_nested_unary
    visit_Cos = remove_nested_unary
    visit_Tanh = remove_nested_unary
    visit_Sigmoid = remove_nested_unary
    visit_Sqrt = remove_nested_unary
    visit_SqrtAbs = remove_nested_unary
    visit_Exp = remove_nested_unary
    visit_Log = remove_nested_unary
    visit_LogAbs = remove_nested_unary

    def visit_Readout(self, node: Readout, *args, **kwargs) -> _Type:
        Readout = self._get_symbol("Readout")
        x = node.operands[0]
        result = yield (x, args, kwargs)

        if kwargs["transform_constant_subtree"] and type(result).__name__ == 'Number':
            Number = self._get_symbol('Number')
            return Number(Readout(result).eval())

        if kwargs["remove_useless_readout"] and result.nettype == "scalar":
            return result

        return Readout(result)

    def visit_Number(self, node: Number, *args, **kwargs) -> _Type:
        yield from yield_nothing()
        return node

    def visit_Variable(self, node: Variable, *args, **kwargs) -> _Type:
        yield from yield_nothing()
        return node

    def visit_Add(self, node: Add, *args, **kwargs) -> _Type:
        Add = self._get_symbol("Add")
        results = []
        for x in node.split_by_add(split_by_sub=True):
            result = yield (x, args, kwargs)
            if type(result).__name__ == 'Number' and result.value == 0:
                continue
            results.append(result)

        if len(results) > 1:
            add = reduce(Add, results)
            results = add.split_by_add(split_by_sub=True, merge_bias=True)
            if (
                len(results) > 1
                and type(results[0]).__name__ == 'Number'
                and results[0].value == 0
            ):
                results = results[1:]

        if len(results) == 1:
            return results[0]

        is_neg = [type(result).__name__ == 'Neg' for result in results]
        neg = [result.operands[0] for result, flag in zip(results, is_neg) if flag]
        pos = [result for result, flag in zip(results, is_neg) if not flag]
        if len(neg) == 0:
            neg = None
        elif len(neg) == 1:
            neg = neg[0]
        else:
            neg = Add(*neg)
        if len(pos) == 0:
            pos = None
        elif len(pos) == 1:
            pos = pos[0]
        else:
            pos = Add(*pos)
        if neg is None:
            return pos
        elif pos is None:
            return -neg
        else:
            return pos - neg

    visit_Sub = visit_Add  # Subtraction is handled the same way as addition

    def visit_Mul(self, node: Mul, *args, **kwargs) -> _Type:
        Mul = self._get_symbol("Mul")
        results = []
        for x in node.split_by_mul(split_by_div=True, merge_coefficients=True):
            result = yield (x, args, kwargs)
            results.append(result)

        if len(results) == 1:
            return results[0]

        is_inv = [type(result).__name__ == 'Inv' for result in results]
        den = [result.operands[0] for result, flag in zip(results, is_inv) if flag]
        num = [result for result, flag in zip(results, is_inv) if not flag]
        if len(den) == 0:
            den = None
        elif len(den) == 1:
            den = den[0]
        else:
            den = reduce(Mul, den)
        if len(num) == 0:
            num = None
        elif len(num) == 1:
            num = num[0]
        else:
            num = reduce(Mul, num)
        if den is None:
            return num
        elif num is None:
            return Inv(den)
        else:
            return num / den

    visit_Div = visit_Mul  # Division is handled the same way as multiplication

    def visit_Neg(self, node: Neg, *args, **kwargs) -> _Type:
        Neg = self._get_symbol("Neg")
        x = node.operands[0]
        result = yield (x, args, kwargs)

        if kwargs["transform_constant_subtree"] and type(result).__name__ == 'Number':
            Number = self._get_symbol('Number')
            return Number(-result.value)

        if type(result).__name__ == 'Neg':
            return result.operands[0]

        if type(result).__name__ == 'Sub':
            return result.operands[1] - result.operands[0]

        if type(result).__name__ == 'Mul' and type(result.operands[0]).__name__ == 'Number':
            result.operands[0].value *= -1
            return result

        if type(result).__name__ == 'Div' and type(result.operands[0]).__name__ == 'Number':
            result.operands[0].value *= -1
            return result

        return Neg(result)

    def visit_Inv(self, node: Inv, *args, **kwargs) -> _Type:
        Inv = self._get_symbol("Inv")
        x = node.operands[0]
        result = yield (x, args, kwargs)

        if kwargs["transform_constant_subtree"] and type(result).__name__ == 'Number':
            Number = self._get_symbol('Number')
            return Number(1 / result.value)

        if type(result).__name__ == 'Inv':
            return result.operands[0]

        if type(result).__name__ == 'Div':
            return result.operands[1] / result.operands[0]

        return Inv(result)

    def visit_Aggr(self, node: Aggr, *args, **kwargs) -> _Type:
        Aggr = self._get_symbol("Aggr")
        x = node.operands[0]
        result = yield (x, args, kwargs)

        # if kwargs["transform_constant_subtree"] and type(result).__name__ == Number:
        #     return Number(Aggr(result).eval())

        if result.nettype == "scalar":
            # D = Number(Aggr(1).eval(), nettype='node')
            D = Aggr(1)
            return D * result

        return Aggr(result)
