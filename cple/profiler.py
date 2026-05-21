from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class ProfileResult:
    output: Any
    runtime_ms: float
    device: str
    sync_mode: str


class OnlineRuntimeProfiler:
    def measure(self, fn: Callable[..., Any], *args: Any, device: str = "cpu", **kwargs: Any) -> ProfileResult:
        if device == "cuda" and torch.cuda.is_available():
            return self._measure_cuda(fn, *args, **kwargs)
        return self._measure_cpu(fn, *args, device=device, **kwargs)

    def _measure_cpu(self, fn: Callable[..., Any], *args: Any, device: str, **kwargs: Any) -> ProfileResult:
        start = time.perf_counter()
        output = fn(*args, **kwargs)
        end = time.perf_counter()
        return ProfileResult(
            output=output,
            runtime_ms=(end - start) * 1000.0,
            device=device,
            sync_mode="wall_clock",
        )

    def _measure_cuda(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> ProfileResult:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        output = fn(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        return ProfileResult(
            output=output,
            runtime_ms=float(start_event.elapsed_time(end_event)),
            device="cuda",
            sync_mode="cuda_event",
        )
