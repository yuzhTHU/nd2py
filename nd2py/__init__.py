# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from .core import *
from . import dataset, generator, search, utils

# 定义模块级别的 __getattr__ 拦截
def __getattr__(name):
    if name == '__all__':
        # Triggered when 'from nd2py import *' is used
        import warnings
        warnings.warn(
            "Detected 'from nd2py import *'. WARNING: This will shadow standard Python built-in functions (such as 'sum').\n"
            "It is strongly recommended to use explicit imports, e.g., 'from nd2py import sum' or 'import nd2py as nd'.",
            category=UserWarning,
            stacklevel=2
        )
        # 动态获取当前模块（__init__.py）中所有不以 '_' 开头的全局变量和模块
        return [n for n in globals() if not n.startswith('_')]

    # 对于其他不存在的属性，保持默认的报错行为
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
