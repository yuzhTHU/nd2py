# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import sys
import importlib
from typing import TYPE_CHECKING, Dict, Tuple

__all__ = ["setup_lazy_imports", "TYPE_CHECKING"]

def setup_lazy_imports(module_name: str, import_mapping: Dict[str, Tuple[str, str]]):
    """Set up lazy imports for a module's ``__init__.py``."""

    def __getattr__(name: str):
        if name in import_mapping:
            module_path, requires = import_mapping[name]
            try:
                module = importlib.import_module(module_path, package=module_name)
                # If the module exposes an attribute with the same name, return it;
                # otherwise the user wants the submodule object itself.
                value = getattr(module, name) if hasattr(module, name) else module
                sys.modules[module_name].__dict__[name] = value
                return value
            except ModuleNotFoundError as e:
                raise ImportError(
                    f"Failed to import '{name}' from '{module_path}' in module "
                    f"'{module_name}' since missing optional dependency. "
                    f"Try to run `pip install nd2py[{requires}]` or "
                    f"`pip install nd2py[all]` to install the required dependencies."
                ) from e
                
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    caller_globals = sys._getframe(1).f_globals
    eager_names = [
        name for name in caller_globals 
        if not name.startswith("_") and 
        name not in {"TYPE_CHECKING", "setup_lazy_imports"}
    ]
    __all__ = sorted(set(eager_names) | import_mapping.keys())

    def __dir__():
        return __all__

    return __getattr__, __dir__, __all__
