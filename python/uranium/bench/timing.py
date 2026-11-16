import ctypes
import math
import statistics
import sys
import time
from dataclasses import dataclass

import torch


@dataclass
class Timing:
    median_ms: float
    p95_ms: float
    mean_ms: float
    sem_ms: float
    reps: int

    def as_dict(self) -> dict:
        return {"median_ms": round(self.median_ms, 4), "p95_ms": round(self.p95_ms, 4), "mean_ms": round(self.mean_ms, 4), "sem_ms": round(self.sem_ms, 4), "reps": self.reps}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, device: torch.device, warmup: int = 3, min_reps: int = 10, max_reps: int = 500, budget_s: float = 10.0, rel_sem: float = 0.01) -> Timing:
    for _ in range(warmup):
        fn()
    sync(device)
    times: list[float] = []
    start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1e3)
        n = len(times)
        if n >= min_reps:
            mean = statistics.fmean(times)
            sem = statistics.stdev(times) / math.sqrt(n)
            if sem <= rel_sem * mean or n >= max_reps or time.perf_counter() - start > budget_s:
                break
    s = sorted(times)
    return Timing(statistics.median(s), s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)], statistics.fmean(s), statistics.stdev(s) / math.sqrt(len(s)), len(s))


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_gib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2**30


def peak_rss_gib() -> float | None:
    if sys.platform.startswith("linux"):
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 2**20
        return None
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_uint32]
        c = Counters()
        c.cb = ctypes.sizeof(c)
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return c.PeakWorkingSetSize / 2**30
    return None
