import threading
import contextlib

__all__ = ["set_fitable", "no_set_fitable"]

_local = threading.local()
_local.set_fitable = True

def set_fitable():
    return getattr(_local, "set_fitable", True)

@contextlib.contextmanager
def no_set_fitable():
    prev = set_fitable()
    _local.set_fitable = False
    try:
        yield
    finally:
        _local.set_fitable = prev
