"""
Lightweight timing utilities for optional performance diagnostics.
"""

import time

__all__ = ["Timer", "AbsTimer", "NamedTimer"]


def time_str(seconds, unit="iter"):
    """Convert seconds to a human-readable string."""
    if seconds < 1e-5:
        return f"{seconds * 1e6:.1f}us/{unit}"
    elif seconds < 1e-4:
        return f"{seconds * 1e6:.0f}us/{unit}"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.0f}us/{unit}"
    elif seconds < 1e-2:
        return f"{seconds * 1e3:.1f}ms/{unit}"
    elif seconds < 1e-1:
        return f"{seconds * 1e3:.0f}ms/{unit}"
    elif seconds < 1:
        return f"{seconds * 1e3:.0f}ms/{unit}"
    elif seconds < 10:
        return f"{seconds:.1f}s/{unit}"
    elif seconds < 60:
        return f"{seconds:.0f}s/{unit}"
    elif seconds < 600:
        return f"{seconds / 60:.1f}min/{unit}"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}min/{unit}"
    elif seconds < 36000:
        return f"{seconds / 3600:.1f}h/{unit}"
    else:
        return f"{seconds / 3600:.0f}h/{unit}"


def speed_str(iter_per_second, unit="iter"):
    if iter_per_second > 1e6:
        return f"{iter_per_second / 1e6:.1f}M{unit}/s"
    elif iter_per_second > 1e3:
        return f"{iter_per_second / 1e3:.1f}K{unit}/s"
    elif iter_per_second > 1:
        return f"{iter_per_second:.1f}{unit}/s"
    elif iter_per_second > 1 / 60:
        return f"{iter_per_second * 60:.1f}{unit}/min"
    elif iter_per_second > 1 / 3600:
        return f"{iter_per_second * 3600:.1f}{unit}/h"
    else:
        return f"{iter_per_second * 86400:.1f}{unit}/day"


class Timer:
    def __init__(self, unit="iter", unit_time=False):
        self.count = 0
        self.time = 0
        self.unit = unit
        self.unit_time = unit_time
        self.start_time = time.time()

    def __str__(self):
        speed = self.speed()
        if self.unit_time:
            return speed_str(speed, self.unit)
        else:
            return time_str(1 / speed, self.unit)

    def add(self, n=1):
        self.count += n
        self.time += time.time() - self.start_time
        self.start_time = time.time()

    def clear(self, reset=False):
        self.count = 0
        self.time = 0
        if reset:
            self.start_time = time.time()

    def speed(self):
        if self.count == 0:
            return 0
        return self.count / self.time


class AbsTimer(Timer):
    def __init__(self, unit="iter"):
        super().__init__(unit=unit)
        self.last = 0

    def add(self, n=1):
        self.count += n - self.last
        self.last = n
        self.time += time.time() - self.start_time
        self.start_time = time.time()


class NamedTimer(Timer):
    def __init__(self, unit="iter"):
        super().__init__(unit=unit)
        self.time = {}
        self.count = {}

    def __str__(self):
        total_time = self.total_time()
        details = []
        for k in sorted(self.time, key=self.time.get, reverse=True):
            t = self.time[k]
            c = self.count[k]
            details.append(f"{k}={c}*{time_str(t/c)}[{t/total_time:.0%}]")
        return f'{time_str(total_time)} ({";".join(details)})'

    def add(self, name, n=1):
        if name not in self.time:
            self.time[name] = self.count[name] = 0
        self.time[name] += time.time() - self.start_time
        self.count[name] += n
        self.start_time = time.time()

    def clear(self, reset=True):
        self.count = {}
        self.time = {}
        if reset:
            self.start_time = time.time()

    def total_time(self):
        return sum(self.time.values())
