import threading
import contextlib

__all__ = ["warn_once", "no_warn"]

_local = threading.local()
_local.set_fitable = True

def warn_once(warn_name, maxsize=None):
    """This function is used to limit the number of times a warning is issued"""
    
    if maxsize is None:
        if getattr(_local, "suppress_warnings", False):
            maxsize = 0
        else:
            maxsize = 1
    
    if not hasattr(warn_once, warn_name):
        setattr(warn_once, warn_name, 0)
    else:
        setattr(warn_once, warn_name, getattr(warn_once, warn_name) + 1)
    return getattr(warn_once, warn_name) < maxsize

@contextlib.contextmanager
def no_warn():
    """Context manager to suppress warnings temporarily"""
    old_value = getattr(_local, "suppress_warnings", False)
    _local.suppress_warnings = True
    try:
        yield
    finally:
        _local.suppress_warnings = old_value