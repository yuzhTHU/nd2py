# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import re
import numbers
import numpy as np
from typing import Generator, Tuple, Dict, List, TYPE_CHECKING
from ..base_visitor import Visitor, yield_nothing
if TYPE_CHECKING:
    from ..symbols import *
    _YieldType = Tuple[Symbol, Tuple, Dict]  # (node, args, kwargs)
    _SendType = str
    _ReturnType = str
    _Type = Generator[_YieldType, _SendType, _ReturnType]


GREEK_LETTERS = [
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 
    'zeta', 'eta', 'theta', 'iota', 'kappa', 
    'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 
    'rho', 'sigma', 'tau', 'upsilon', 
    'phi', 'chi', 'psi', 'omega'
]
def replace_greek(s):
    def repl(m):
        match = m.group(0)
        if match.lower() in GREEK_LETTERS:
            return rf"\{match.lower()}"
        return match
    return re.sub(r'\b[a-zA-Z]+\b', repl, s)


class StringPrinter(Visitor):
    def __call__(
        self,
        node: Symbol,
        raw=False,
        latex=False,
        number_format="",
        omit_mul_sign=False,
        skeleton=False,
        grouped_parameter_symbol='alpha',
    ) -> str:
        """
        Args:
        - raw:bool=False, whether to return the raw format
        - number_format:str='', can be '0.2f'
        - omit_mul_sign:bool=False, whether to omit the multiplication sign
        - latex:bool=False, whether to return the latex format
        - skeleton:bool=False, whether to ignore the concrete values of Number
        """

        grouped_parameters = []
        for item in node.iter_preorder():
            if type(item).__name__ == "GroupedParameter":
                grouped_parameters.append(item)

        return super().__call__(
            node,
            raw=raw,
            latex=latex,
            number_format=number_format,
            omit_mul_sign=omit_mul_sign,
            skeleton=skeleton,
            grouped_parameters=grouped_parameters,
            grouped_parameter_symbol=grouped_parameter_symbol,
        )

    def generic_visit(self, node: Symbol, *args, **kwargs) -> _Type:
        yield from yield_nothing()
        name = type(node).__name__
        if not kwargs.get("raw"):
            name = name.lower()
        children = []
        for x in node.operands:
            child = yield (x, args, kwargs)
            children.append(child)
        return f"{name}({', '.join(children)})"

    def visit_Empty(self, node: Empty, *args, **kwargs) -> _Type:
        yield from yield_nothing()
        if kwargs.get("raw", False):
            return "Empty()"
        if kwargs.get("latex", False):
            return r"\square"
        return "?"

    def visit_Number(self, node: Number, *args, **kwargs) -> _Type:
        yield from yield_nothing()
        if kwargs.get("raw", False):
            return f'Number({np.array(node.value).tolist()}, "{node.nettype}", {node.fitable})'
        if kwargs.get("skeleton", False):
            if node.nettype == "scalar":
                return rf"C" if kwargs.get("latex") else "C"
            elif node.nettype == "node":
                return rf"C_n" if kwargs.get("latex") else f"Cn"
            elif node.nettype == "edge":
                return rf"C_e" if kwargs.get("latex") else f"Ce"
            else:
                raise ValueError(
                    f"Unknown nettype: {node.nettype}. Expected 'scalar', 'node', or 'edge'."
                )
        if type(node.value).__module__.startswith("torch"):
            content = np.array(node.value.tolist())
        else:
            content = np.array(node.value)
        fmt = kwargs.get("number_format", "")
        if isinstance(content, (numbers.Number)) or content.size == 1:
            try:
                if np.ndim(content) > 0:
                    content = content[0]
                if int(content) == content:
                    content = str(int(content))
                else:
                    content = f"{content:{fmt}}"
            except:
                content = str(content)
        elif kwargs.get("latex", False):
            content = rf"\left<{np.mean(content):{fmt}}\right>"
        else:
            content = f"<{np.mean(content):{fmt}}>"  #  (+{np.std(content):{fmt}})
        return content if node.fitable else f"Constant({content})"

    def visit_Variable(self, node: Variable, *args, **kwargs) -> _Type:
        yield from yield_nothing()
        if kwargs.get("raw", False):
            return f'Variable("{node.name}", "{node.nettype}")'
        if kwargs.get("latex", False):
            if len(node.name) == 1:
                return node.name
            elif node.name.lower() in GREEK_LETTERS:
                return replace_greek(node.name)
            elif '_' in node.name:
                var, suffix = node.name.split('_', 1)
                var = replace_greek(var)
                suffix = replace_greek(suffix.replace('_', ' '))
                return f"{var}_{{{suffix}}}"
            else: # frequency -> f_{frequency}, lambda -> l_{lambda}
                var, suffix = node.name[0], node.name
                suffix = replace_greek(suffix)
                return f"{var}_{{{suffix}}}"
        return node.name

    def visit_GroupedParameter(self, node: GroupedParameter, *args, **kwargs) -> _Type:
        yield from yield_nothing()
        by = yield (node.by, args, kwargs)
        if kwargs.get("raw", False):
            return (
                f"GroupedParameter({by}, "
                f"value={node.value_dict!r}, default={node.default!r}, "
                f"fitable={node.fitable!r}, nettype={node.nettype!r})"
            )
        sym = kwargs.get("grouped_parameter_symbol", "alpha")
        if len(grouped_parameters := kwargs.get("grouped_parameters", [])) > 1:
            assert node in grouped_parameters, f"GroupedParameter {node} not found in grouped_parameters list."
            sym_idx = grouped_parameters.index(node) + 1
        else:
            sym_idx = None
        if kwargs.get("latex"):
            sym = replace_greek(sym)
            return rf"{sym}^{{({sym_idx})}}_{{{by}}}" if sym_idx is not None else rf"{sym}_{{{by}}}"
        return f"{sym}^({sym_idx})[{by}]" if sym_idx is not None else f"{sym}[{by}]"

    def visit_Add(self, node: Add, *args, **kwargs) -> _Type:
        x1 = yield (node.operands[0], args, kwargs)
        x2 = yield (node.operands[1], args, kwargs)
        return f"{x1} + {x2}"

    def visit_Sub(self, node: Sub, *args, **kwargs) -> _Type:
        x1 = yield (node.operands[0], args, kwargs)
        x2 = yield (node.operands[1], args, kwargs)
        if type(node.operands[1]).__name__ in ['Add', 'Sub']:
            x2 = (
                rf"\left({x2}\right)"
                if kwargs.get("latex", False) and r"\left" in x2
                else f"({x2})"
            )
        return f"{x1} - {x2}"

    def visit_Mul(self, node: Mul, *args, **kwargs) -> _Type:
        x1 = yield (node.operands[0], args, kwargs)
        x2 = yield (node.operands[1], args, kwargs)
        if type(node.operands[0]).__name__ in ['Add', 'Sub']:
            x1 = (
                rf"\left({x1}\right)"
                if kwargs.get("latex", False) and r"\left" in x1
                else f"({x1})"
            )
        if type(node.operands[1]).__name__ in ['Add', 'Sub']:
            x2 = (
                rf"\left({x2}\right)"
                if kwargs.get("latex", False) and r"\left" in x2
                else f"({x2})"
            )
        if kwargs.get("omit_mul_sign", False):
            if type(node.operands[1]).__name__ in ['Add', 'Sub']:
                return f"{x1}{x2}"
            if isinstance(node.operands[0], Number) and isinstance(
                node.operands[1], Variable
            ):
                return f"{x1}{x2}"
        return (
            f"{x1} * {x2}" if not kwargs.get("latex", False) else rf"{x1} \times {x2}"
        )

    def visit_Div(self, node: Div, *args, **kwargs) -> _Type:
        x1 = yield (node.operands[0], args, kwargs)
        x2 = yield (node.operands[1], args, kwargs)
        if kwargs.get("latex", False):
            return rf"\frac{{{x1}}}{{{x2}}}"
        if type(node.operands[0]).__name__ in ['Add', 'Sub']:
            x1 = (
                rf"\left({x1}\right)"
                if kwargs.get("latex", False) and r"\left" in x1
                else f"({x1})"
            )
        if type(node.operands[1]).__name__ in ['Add', 'Sub', 'Mul', 'Div', 'Inv']:
            x2 = (
                rf"\left({x2}\right)"
                if kwargs.get("latex", False) and r"\left" in x2
                else f"({x2})"
            )
        return f"{x1} / {x2}"

    def visit_Pow(self, node: Pow, *args, **kwargs) -> _Type:
        x1 = yield (node.operands[0], args, kwargs)
        x2 = yield (node.operands[1], args, kwargs)
        if type(node.operands[0]).__name__ in [
            'Add', 'Sub', 'Mul', 'Div', 'Pow', 'Neg', 'Inv', 'Pow2', 'Pow3',
        ]:
            x1 = (
                rf"\left({x1}\right)"
                if kwargs.get("latex", False) and r"\left" in x1
                else f"({x1})"
            )
        if kwargs.get("latex", False):
            return rf"{x1}^{{{x2}}}"
        if type(node.operands[1]).__name__ in ['Add', 'Sub', 'Mul', 'Div', 'Inv']:
            x2 = (
                rf"\left({x2}\right)"
                if kwargs.get("latex", False) and r"\left" in x2
                else f"({x2})"
            )
        return f"{x1} ** {x2}"

    def visit_Neg(self, node: Neg, *args, **kwargs) -> _Type:
        x = yield (node.operands[0], args, kwargs)
        if type(node.operands[0]).__name__ in ['Add', 'Sub']:
            x = f"({x})"
        return f"-{x}"

    def visit_Inv(self, node: Inv, *args, **kwargs) -> _Type:
        x = yield (node.operands[0], args, kwargs)
        if kwargs.get("latex", False):
            return rf"\frac{{1}}{{{x}}}"
        if type(node.operands[0]).__name__ in ['Add', 'Sub', 'Mul', 'Div']:
            x = f"({x})"
        return f"1 / {x}"

    def visit_Pow2(self, node: Pow2, *args, **kwargs) -> _Type:
        x = yield (node.operands[0], args, kwargs)
        if type(node.operands[0]).__name__ in [
            'Add', 'Sub', 'Mul', 'Div', 'Pow', 'Neg', 'Inv', 'Pow2', 'Pow3',
        ]:
            x = f"({x})"
        return f"{x} ** 2" if not kwargs.get("latex", False) else f"{x}^2"

    def visit_Pow3(self, node: Pow3, *args, **kwargs) -> _Type:
        x = yield (node.operands[0], args, kwargs)
        if type(node.operands[0]).__name__ in [
            'Add', 'Sub', 'Mul', 'Div', 'Pow', 'Neg', 'Inv', 'Pow2', 'Pow3',
        ]:
            x = f"({x})"
        return f"{x} ** 3" if not kwargs.get("latex", False) else f"{x}^3"

    def visit_Sour(self, node: Sour, *args, **kwargs) -> _Type:
        x = yield (node.operands[0], args, kwargs)
        if kwargs.get("latex"):
            return rf"\phi_s\left({x}\right)" if r"\left" in x else rf"\phi_s({x})"
        if kwargs.get("raw"):
            return f"{type(node).__name__}({x})"
        return f"{type(node).__name__.lower()}({x})"

    def visit_Targ(self, node: Targ, *args, **kwargs) -> _Type:
        x = yield (node.operands[0], args, kwargs)
        if kwargs.get("latex"):
            return rf"\phi_t\left({x}\right)" if r"\left" in x else rf"\phi_t({x})"
        if kwargs.get("raw"):
            return f"{type(node).__name__}({x})"
        return f"{type(node).__name__.lower()}({x})"

    def visit_Aggr(self, node: Aggr, *args, **kwargs) -> _Type:
        x = yield (node.operands[0], args, kwargs)
        if kwargs.get("latex"):
            return rf"\rho\left({x}\right)" if r"\left" in x else rf"\rho({x})"
        if kwargs.get("raw"):
            return f"{type(node).__name__}({x})"
        return f"{type(node).__name__.lower()}({x})"

    def visit_Rgga(self, node: Rgga, *args, **kwargs) -> _Type:
        x = yield (node.operands[0], args, kwargs)
        if kwargs.get("latex"):
            return (
                rf"\rho^{-1}\left({x}\right)" if r"\left" in x else rf"\rho^{-1}({x})"
            )
        if kwargs.get("raw"):
            return f"{type(node).__name__}({x})"
        return f"{type(node).__name__.lower()}({x})"
