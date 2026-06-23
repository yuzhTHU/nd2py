# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import core, dataset, generator, search, utils
    from .core import *

_SUBMODULES = ("core", "dataset", "generator", "search", "utils")


def _load_module(name: str):
    module = globals()[name] = importlib.import_module(f".{name}", __name__)
    return module


def __getattr__(name: str):
    if name == "__all__":
        import warnings
        warnings.warn(
            "Using `from nd2py import *` is strongly discouraged as "
            "it overrides Python built-in functions like `max`, `sum`, etc. "
            "Please use `import nd2py as nd` instead.",
            UserWarning,
            stacklevel=2  # stacklevel=2 让警告指向用户的调用代码，而不是 __init__.py
        )
        return __dir__()
    elif name in _SUBMODULES:
        return _load_module(name)
    elif name in dir(core_module := globals().get("core") or _load_module("core")):
        value = globals()[name] = getattr(core_module, name)
        return value
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    names = {name for name in globals() if not name.startswith("_") and name != "TYPE_CHECKING"}
    names.update(_SUBMODULES)
    core_module = globals().get("core") or _load_module("core")
    names.update(name for name in dir(core_module) if not name.startswith("_"))
    return sorted(names)


# Do not define __all__; this keeps `from nd2py import *` routed through the warning above.
