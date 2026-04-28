"""Profiling utilities for lmlib backends.

Collects wall time, peak memory, and sub-call statistics for decorated
functions. Profiling is off by default and adds zero overhead unless enabled.

Usage
-----
>>> import lmlib as lm
>>> lm.profiling.enable()
>>> # ... run your computation ...
>>> lm.profiling.report()
>>> lm.profiling.disable()
>>> lm.profiling.clear()
"""

import cProfile
import io
import pstats
import time
import tracemalloc
from functools import wraps

try:
    import psutil as _psutil
    _psutil.cpu_percent(interval=None)  # prime the rolling counter (first call always returns 0.0)
    _HAS_PSUTIL = True
except ImportError:
    _psutil = None
    _HAS_PSUTIL = False

_enabled: bool = False
_store: dict = {}

# Records with cpu_percent above this threshold are considered unreliable.
CPU_WARN_THRESHOLD: float = 50.0


class _Record:
    __slots__ = ("elapsed_s", "peak_bytes", "_pstats", "cpu_percent", "load_1m")

    def __init__(self, elapsed_s: float, peak_bytes: int, ps,
                 cpu_percent: float = float('nan'), load_1m: float = float('nan')):
        self.elapsed_s = elapsed_s
        self.peak_bytes = peak_bytes
        self._pstats = ps
        self.cpu_percent = cpu_percent   # system-wide CPU % at call start (nan if psutil unavailable)
        self.load_1m = load_1m           # 1-minute load average at call start (nan if unavailable)

    @property
    def is_reliable(self) -> bool:
        """False if CPU occupancy exceeded CPU_WARN_THRESHOLD at the time of the call."""
        import math
        return math.isnan(self.cpu_percent) or self.cpu_percent < CPU_WARN_THRESHOLD

    def __repr__(self) -> str:
        flag = "" if self.is_reliable else "  !HIGH-CPU"
        return (
            f"<ProfileRecord "
            f"time={self.elapsed_s * 1e3:.2f}ms "
            f"peak={self.peak_bytes / 1024:.1f}KB "
            f"cpu={self.cpu_percent:.0f}%{flag}>"
        )


def enable() -> None:
    """Enable profiling. Decorated functions will collect data after this call."""
    global _enabled
    _enabled = True


def disable() -> None:
    """Disable profiling. Decorated functions pass through with zero overhead."""
    global _enabled
    _enabled = False


def clear() -> None:
    """Discard all stored profiling records."""
    _store.clear()


def get_records() -> dict:
    """Return the raw ``{func_name: [_Record, ...]}`` dict."""
    return _store


def get_reliable_records(threshold: float = None) -> dict:
    """Return only records where CPU occupancy was below *threshold* percent.

    Parameters
    ----------
    threshold : float, optional
        CPU % cutoff.  Defaults to ``lm.profiling.CPU_WARN_THRESHOLD``.
        Records collected without psutil are always included.
    """
    import math
    t = CPU_WARN_THRESHOLD if threshold is None else threshold
    return {
        name: [r for r in recs if math.isnan(r.cpu_percent) or r.cpu_percent < t]
        for name, recs in _store.items()
    }


def report(top: int = 15) -> None:
    """Print a human-readable profiling summary to stdout.

    Parameters
    ----------
    top : int
        Number of sub-calls to show per function in the cProfile breakdown.
    """
    if not _store:
        print("No profiling data collected. Call `lm.profiling.enable()` first.")
        return

    import math
    for name, records in _store.items():
        n = len(records)
        noisy = [r for r in records if not math.isnan(r.cpu_percent) and r.cpu_percent >= CPU_WARN_THRESHOLD]
        reliable = [r for r in records if math.isnan(r.cpu_percent) or r.cpu_percent < CPU_WARN_THRESHOLD]
        use = reliable if reliable else records  # fall back to all if everything was noisy

        avg_ms = sum(r.elapsed_s for r in use) / len(use) * 1e3
        max_ms = max(r.elapsed_s for r in use) * 1e3
        avg_kb = sum(r.peak_bytes for r in use) / len(use) / 1024
        max_kb = max(r.peak_bytes for r in use) / 1024

        print(f"\n{'─' * 68}")
        print(f"  {name}")
        noisy_note = f"  ⚠ {len(noisy)}/{n} calls excluded (cpu >= {CPU_WARN_THRESHOLD:.0f}%)" if noisy else ""
        print(
            f"  calls={len(use)}/{n}{noisy_note}  "
            f"avg={avg_ms:.2f} ms  max={max_ms:.2f} ms  "
            f"avg_peak={avg_kb:.1f} KB  max_peak={max_kb:.1f} KB"
        )
        if not _HAS_PSUTIL:
            print("  (install psutil for CPU-occupancy filtering)")
        print(f"{'─' * 68}")

        last = records[-1]
        if last._pstats is not None:
            last._pstats.sort_stats("cumulative")
            last._pstats.print_stats(top)


def profile(func):
    """Decorator: collect wall time, peak memory, and sub-call stats when enabled.

    When profiling is disabled (the default) the decorator is a transparent
    pass-through with no measurable overhead.

    Profile data is accumulated in ``lm.profiling.get_records()`` and printed
    with ``lm.profiling.report()``.

    Examples
    --------
    >>> from lmlib.utils.profiling import profile
    >>> @profile
    ... def my_func(x):
    ...     return x * 2
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _enabled:
            return func(*args, **kwargs)

        # Avoid stopping an outer tracemalloc session (nested @profile calls).
        already_tracing = tracemalloc.is_tracing()
        if not already_tracing:
            tracemalloc.start()

        # Sample CPU occupancy just before the call so we can flag noisy measurements.
        if _HAS_PSUTIL:
            cpu_pct = _psutil.cpu_percent(interval=None)  # non-blocking, uses rolling interval
            load_1m = _psutil.getloadavg()[0] if hasattr(_psutil, 'getloadavg') else float('nan')
        else:
            cpu_pct = float('nan')
            load_1m = float('nan')

        prof = cProfile.Profile()
        t0 = time.perf_counter()
        result = prof.runcall(func, *args, **kwargs)
        elapsed = time.perf_counter() - t0

        _, peak = tracemalloc.get_traced_memory()
        if not already_tracing:
            tracemalloc.stop()

        buf = io.StringIO()
        ps = pstats.Stats(prof, stream=buf)
        _store.setdefault(func.__name__, []).append(
            _Record(elapsed, peak, ps, cpu_percent=cpu_pct, load_1m=load_1m)
        )

        return result

    return wrapper
