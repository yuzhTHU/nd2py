# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from typing import TYPE_CHECKING
from . import core, dataset, generator, search, utils

# 扁平化 core 中模块的导入，提供更简洁的接口
if TYPE_CHECKING:
    from .core import *

from . import core
_export_modules = [core]

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

    for mod in _export_modules:
        if name in dir(mod): # 使用 dir 避免触发下层模块的 __getattr__ 导致懒加载失效
            value = getattr(mod, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    exclude_names = {"TYPE_CHECKING", "core"}
    names = {name for name in globals() if not name.startswith("_") and name not in exclude_names}
    for mod in _export_modules:
        names.update([n for n in dir(mod) if not n.startswith("_")])
    return list(names)

# __all__ = __dir__() # 不定义 __all__, 以保证用户使用 from nd2py import * 时触发 __getattr__ 中的警告
