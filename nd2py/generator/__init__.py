# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from importlib import import_module

__all__ = ['data', 'eq'] + [
    *import_module(".data", __name__).__all__,
    *import_module(".eq", __name__).__all__,
]

def __getattr__(name):
    for submodule in ['data', 'eq']:
        module = import_module(f".{submodule}", __name__)
        if name == submodule:
            return module
        if name in dir(module):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
