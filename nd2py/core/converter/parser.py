# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import ast
import importlib
from typing import Dict
from ..symbols import Symbol, Number, Variable

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


# module = importlib.import_module("..symbols", package=__package__)
# default_callables = {name: getattr(module, name) for name in module.__all__}
# module = importlib.import_module("..symbols", package=__package__)
# default_callables |= {name: getattr(module, name) for name in module.__all__}
module = importlib.import_module("..symbols", package=__package__)
default_callables = {
    name: getattr(module, name) 
    for name in dir(module) 
    # 1. 过滤掉以 '_' 开头的私有属性和内置魔法属性（如 __name__, __doc__）
    # 2. 确保获取到的是可调用的对象（类或函数），这能自动过滤掉意外暴露的子模块
    if not name.startswith('_') and callable(getattr(module, name))
}

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