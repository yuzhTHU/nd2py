# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .calc import *
    from .symbols import *
    from .context import *
    from .converter import *
    from .transform import *

from . import calc, symbols, context, converter, transform
_export_modules = [calc, symbols, context, converter, transform]

def __getattr__(name: str):
    for mod in _export_modules:
        if name in dir(mod):
            value = getattr(mod, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    exclude_names = {"TYPE_CHECKING", "calc", "symbols", "context", "converter", "transform"}
    names = {name for name in globals() if not name.startswith("_") and name not in exclude_names}
    for mod in _export_modules:
        names.update([n for n in dir(mod) if not n.startswith("_")])
    return list(names)

__all__ = __dir__()
