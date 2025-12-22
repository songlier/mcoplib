import torch
import statistics
import math
from typing import Callable, List, Optional

def measure_cuda(kernel_fn: Callable[[], None],
                 iters: int = 200,
                 warmup: int = 20,
                 device: Optional[int] = 0,
                 stream: Optional[torch.cuda.Stream] = None) -> dict:
    """
    Measure GPU kernel execution time using CUDA events.
    kernel_fn: a callable that launches GPU work (non-blocking)
    iters: number of measured iterations
    warmup: number of warm-up iterations (not measured)
    device: cuda device index or None to auto
    stream: torch.cuda.Stream to run on (optional). If None, default stream is used.
    Returns stats in microseconds (us): list, mean, median, std, min
    """
    if device is None:
        device = torch.cuda.current_device()
    torch.cuda.set_device(device)

    if stream is None:
        stream = torch.cuda.default_stream(device)

    # Warm-up
    for _ in range(warmup):
        # record + launch + record + sync to ensure real warmup
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        kernel_fn()
        end.record(stream)
        end.synchronize()

    # Measured runs
    times_ms: List[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)        # record start on stream
        kernel_fn()                 # launch kernel (async)
        end.record(stream)          # record end on stream
        end.synchronize()           # wait for this end event (blocks CPU until kernel done)
        ms = start.elapsed_time(end)  # milliseconds (float)
        times_ms.append(ms)

    # convert to microseconds
    times_us = [t * 1000.0 for t in times_ms]

    stats = {
        "times_us": times_us,
        "mean_us": statistics.mean(times_us),
        "median_us": statistics.median(times_us),
        "min_us": min(times_us),
        "max_us": max(times_us),
        "stdev_us": statistics.stdev(times_us) if len(times_us) > 1 else 0.0,
        "iters": iters,
        "warmup": warmup
    }
    return stats