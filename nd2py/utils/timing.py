"""
Lightweight timing utilities for optional performance diagnostics.
"""

import time
from typing import Literal

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


def to_str(count, time, unit, mode):
    if mode == "pace":  # time per count
        return time_str(time / count, unit) if count > 0 else "NaN"
    elif mode == "speed":  # count per time
        return speed_str(count / time, unit) if time > 0 else "NaN"
    elif mode == "counter":  # count
        return speed_str(count, unit).rsplit("/", 1)[0]
    elif mode == "timer":  # time
        return time_str(time, "").rsplit("/", 1)[0]


class Timer:
    def __init__(
        self, unit="iter", mode: Literal["pace", "speed", "counter", "timer"] = "pace"
    ):
        self.count = 0
        self.time = 0
        self.unit = unit
        self.mode = mode
        self.start_time = time.time()

    def __str__(self):
        return to_str(self.count, self.time, self.unit, self.mode)

    def add(self, n=1):
        self.count += n
        self.time += time.time() - self.start_time
        self.start_time = time.time()

    def clear(self, reset=False):
        self.count = 0
        self.time = 0
        if reset:
            self.start_time = time.time()

    @property
    def speed(self):
        if self.count == 0:
            return 0
        return self.count / self.time


class AbsTimer(Timer):
    def __init__(
        self, unit="iter", mode: Literal["pace", "speed", "counter", "timer"] = "pace"
    ):
        super().__init__(unit=unit, mode=mode)
        self.start_count = 0

    def add(self, n=1):
        self.count += n - self.start_count
        self.start_count = n
        self.time += time.time() - self.start_time
        self.start_time = time.time()

    def clear(self, reset=False):
        self.count = 0
        self.time = 0
        if reset:
            self.start_count = 0
            self.start_time = time.time()


class NamedTimer(Timer):
    def __init__(
        self, unit="iter", mode: Literal["pace", "speed", "counter", "timer"] = "pace"
    ):
        self._count = {}
        self._time = {}
        self.unit = unit
        self.mode = mode
        self.start_time = time.time()

    def __str__(self):
        msg_list = {}
        pct_list = {}
        if not self._time:
            return 'None'
        for k in self._time:
            msg_list[k] = to_str(self._count[k], self._time[k], self.unit, self.mode)
            if self.mode == "pace":
                pct_list[k] = self._time[k] / self.time if self.time > 0 else 0
                prefix = to_str(self.count, self.time, self.unit, "timer")
            elif self.mode == "speed":
                pct_list[k] = self._count[k] / self.count if self.count > 0 else 0
                prefix = to_str(self.count, self.time, self.unit, "counter")
            elif self.mode == "counter":
                pct_list[k] = self._time[k] / self.time if self.time > 0 else 0
                prefix = to_str(self.count, self.time, self.unit, "counter")
            elif self.mode == "timer":
                pct_list[k] = self._count[k] / self.count if self.count > 0 else 0
                prefix = to_str(self.count, self.time, self.unit, "timer")
        detail = []
        for k in sorted(self._time.keys(), key=pct_list.get, reverse=True):
            detail.append(f"{k}={msg_list[k]}[{pct_list[k]:.0%}]")
        return f'{prefix} ({"; ".join(detail)})'

    def add(self, name, n=1, update_time=True):
        if name not in self._time:
            self._time[name] = self._count[name] = 0
        self._time[name] += time.time() - self.start_time
        self._count[name] += n
        if update_time:
            self.start_time = time.time()

    def clear(self, reset=True):
        self._count = {}
        self._time = {}
        if reset:
            self.start_time = time.time()

    def total_time(self):
        return self.time

    @property
    def time(self):
        return sum(self._time.values())

    @property
    def count(self):
        return sum(self._count.values())
