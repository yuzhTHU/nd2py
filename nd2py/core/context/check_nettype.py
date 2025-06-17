import threading
import contextlib

__all__ = ["check_nettype", "no_nettype_check"]

_local = threading.local()
_local.check_nettype = True

def check_nettype():
    return getattr(_local, "check_nettype", True)

@contextlib.contextmanager
def no_nettype_check():
    prev = check_nettype()
    _local.check_nettype = False
    try:
        yield
    finally:
        _local.check_nettype = prev
