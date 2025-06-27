import torch
import numbers
import numpy as np
from typing import Generator, Tuple, Dict, List
from ..symbols import *
from ..base_visitor import Visitor, yield_nothing

_YieldType = Tuple[Symbol, Tuple, Dict]  # (node, args, kwargs)
_SendType = str
_ReturnType = str
_Type = Generator[_YieldType, _SendType, _ReturnType]


class TreePrinter(Visitor):
    def __call__(
        self, node: Symbol, number_format="", flat=False, skeleton=False
    ) -> str:
        """
        Args:
        - number_format:str='', can be '0.2f'
        - flat:bool=False, whether to flat the Add and Mul
        - omit_mul_sign:bool=False, whether to omit the multiplication sign
        """

        return super().__call__(
            node, number_format=number_format, flat=flat, skeleton=skeleton
        )

    def generic_visit(self, node: Symbol, *args, **kwargs) -> _Type:
        yield from yield_nothing()
        name = f"{type(node).__name__} ({node.nettype})"
        children = []
        for op in node.operands:
            child = yield (op, args, kwargs)
            children.append(child)
        for idx, child in enumerate(children):
            children[idx] = ("├ " if idx < len(children) - 1 else "└ ") + child.replace(
                "\n", "\n" + ("┆ " if idx < len(children) - 1 else "  ")
            )
        return name + "\n" + "\n".join(children)

    def visit_Empty(self, node: Symbol, *args, **kwargs) -> _Type:
        yield from yield_nothing()
        return f"? ({node.nettype})"

    def visit_Number(self, node: Number, *args, **kwargs) -> _Type:
        yield from yield_nothing()
        if kwargs.get("skeleton", False):
            return f"C ({node.nettype})"
        if isinstance(node.value, torch.Tensor):
            content = np.array(node.value.tolist())
        else:
            content = np.array(node.value)
        fmt = kwargs.get("number_format", "")
        if isinstance(content, (numbers.Number)) or content.size == 1:
            content = f"{content:{fmt}}"
        else:
            content = f"<{np.mean(content):{fmt}}>"  #  (+{np.std(content):{fmt}})
        return (
            content + f" ({node.nettype})"
            if node.fitable
            else f"Constant({content}) ({node.nettype})"
        )

    def visit_Variable(self, node: Variable, *args, **kwargs) -> _Type:
        yield from yield_nothing()
        return f"{node.name} ({node.nettype})"

    def visit_Add(self, node: Add, *args, **kwargs) -> _Type:
        if kwargs.get("flat", False):
            name = f"{type(node).__name__} ({node.nettype})"
            children = []
            for op in node.split_by_add():
                child = yield (op, args, kwargs)
                children.append(child)
            for idx, child in enumerate(children):
                children[idx] = (
                    "├ " if idx < len(children) - 1 else "└ "
                ) + child.replace(
                    "\n", "\n" + ("┆ " if idx < len(children) - 1 else "  ")
                )
            return name + "\n" + "\n".join(children)
        return (yield from self.generic_visit(node, *args, **kwargs))

    def visit_Mul(self, node: Mul, *args, **kwargs) -> _Type:
        if kwargs.get("flat", False):
            name = f"{type(node).__name__} ({node.nettype})"
            children = []
            for op in node.split_by_mul():
                child = yield (op, args, kwargs)
                children.append(child)
            for idx, child in enumerate(children):
                children[idx] = (
                    "├ " if idx < len(children) - 1 else "└ "
                ) + child.replace(
                    "\n", "\n" + ("┆ " if idx < len(children) - 1 else "  ")
                )
            return name + "\n" + "\n".join(children)
        return (yield from self.generic_visit(node, *args, **kwargs))
