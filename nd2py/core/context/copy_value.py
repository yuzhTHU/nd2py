# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import threading
import contextlib

__all__ = ["get_copy_value", "set_copy_value", "no_copy_value"]

_local = threading.local()
_local.copy_value = True

def get_copy_value():
    return getattr(_local, "copy_value", True)

def set_copy_value(value: bool):
    _local.copy_value = value

@contextlib.contextmanager
def no_copy_value():
    prev = get_copy_value()
    set_copy_value(False)
    try:
        yield
    finally:
        set_copy_value(prev)