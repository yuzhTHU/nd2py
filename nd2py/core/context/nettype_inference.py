# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import threading
import contextlib

__all__ = ["nettype_inference", "no_nettype_inference", "set_nettype_inference"]

_local = threading.local()
_local.nettype_inference = True

def nettype_inference():
    return getattr(_local, "nettype_inference", True)

@contextlib.contextmanager
def no_nettype_inference():
    prev = nettype_inference()
    set_nettype_inference(False)
    try:
        yield
    finally:
        set_nettype_inference(prev)

def set_nettype_inference(value: bool):
    _local.nettype_inference = value
