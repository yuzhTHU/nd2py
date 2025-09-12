import ast
import importlib
from typing import Dict
from ..symbols import Symbol, Number, Variable
from ..functions import *

__all__ = ["parse"]


def get_variables_and_callables(code: str):
    """
    "x + exp(y) * sin(z) + 1 - pi" -> ({"x", "y", "z", "pi"}, {"exp", "sin"})
    """
    tree = ast.parse(code, mode="eval")
    callables = set()
    variables = set()

    class Analyzer(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                callables.add(node.func.id)
            self.generic_visit(node)

        def visit_Name(self, node):
            variables.add(node.id)

    Analyzer().visit(tree)
    variables -= callables
    return variables, callables


module = importlib.import_module("..functions", package=__package__)
default_callables = {name: getattr(module, name) for name in module.__all__}


def parse(
    expression: str,
    variables: Dict[str, Symbol] = None,
    callables: Dict[str, callable] = None,
) -> Symbol:
    _variables, _callables = get_variables_and_callables(expression)
    default_variables = {name: Variable(name, nettype="scalar") for name in _variables}
    if variables is None:
        variables = default_variables
    else:
        variables = {**default_variables, **variables}
    if callables is None:
        callables = default_callables
    else:
        callables = {**default_callables, **callables}
    # Check if all required variables and callables are provided
    missing_calls = _callables - callables.keys()
    if missing_calls:
        raise ValueError(f"Undefined callables: {missing_calls}")
    # Check for name conflicts
    duplicate_keys = variables.keys() & callables.keys()
    if duplicate_keys:
        # raise ValueError(
        #     f"Name conflict between variables and callables: {duplicate_keys}"
        # )
        Warning(f"Name conflict between variables and callables: {duplicate_keys}")
    result = eval(expression, {}, {**callables, **variables})
    if isinstance(result, (int, float)):
        result = Number(result)
    return result