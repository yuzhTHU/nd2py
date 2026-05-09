# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import sys
import importlib
from typing import Dict, Tuple, TYPE_CHECKING

__all__ = ["setup_lazy_imports", "TYPE_CHECKING"]


def setup_lazy_imports(module_name: str, import_mapping: Dict[str, Tuple[str, str]]):
    """Set up lazy imports for a module's ``__init__.py``.

    Returns ``(__getattr__, __dir__, __all__)`` which should be assigned at
    the module level so that ``from package import OptionalClass`` works
    without importing the optional dependency until it is actually needed.

    Args:
        module_name: The ``__name__`` of the calling module.
        import_mapping: A dict mapping attribute names to
            ``(module_path, requires)`` tuples.  *module_path* is a
            relative import path (e.g. ``".torch_calc"``) and *requires*
            is the optional-dependency group name (e.g. ``"nn"``) shown in
            the error message when the dependency is missing.

    Usage::

        # __init__.py
        from .core import CoreClass
        from ..utils.lazy_loader import setup_lazy_imports

        if TYPE_CHECKING:
            from .optional import OptionalClass

        __getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
            "OptionalClass": (".optional", "nn"),
        })
    """

    def __getattr__(name: str):
        if name in import_mapping:
            module_path, requires = import_mapping[name]
            try:
                module = importlib.import_module(module_path, package=module_name)
                # If the module exposes an attribute with the same name, return it;
                # otherwise the user wants the submodule object itself.
                return getattr(module, name) if hasattr(module, name) else module
            except ImportError as e:
                raise ImportError(
                    f"Failed to import '{name}' from '{module_path}' in module "
                    f"'{module_name}' since missing optional dependency. "
                    f"Try to run `pip install nd2py[{requires}]` or "
                    f"`pip install nd2py[all]` to install the required dependencies."
                ) from e

        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    # Build __all__ from the caller's existing globals plus the lazy names.
    caller_globals = sys._getframe(1).f_globals
    eager_names = [name for name in caller_globals if not name.startswith("_")]
    __all__ = sorted(set(eager_names) | import_mapping.keys())

    def __dir__():
        return __all__

    return __getattr__, __dir__, __all__
