from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .adapters import MockSionnaAdapter, SionnaStepResult
from .config import MockAdapterConfig


@dataclass
class SionnaEnvironmentInfo:
    available: bool
    version: str | None
    torch_version: str
    cuda_available: bool
    note: str


def inspect_sionna_environment() -> SionnaEnvironmentInfo:
    try:
        import sionna

        return SionnaEnvironmentInfo(
            available=True,
            version=getattr(sionna, "__version__", "unknown"),
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            note="Sionna is importable. CUDA is optional for CPLE CPU smoke runs.",
        )
    except Exception as exc:  # pragma: no cover
        return SionnaEnvironmentInfo(
            available=False,
            version=None,
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            note=f"Sionna import failed: {exc}",
        )


class SionnaAdapter:
    """Thin boundary for future real Sionna SYS integration.

    The first implementation keeps the public step API stable and can wrap a
    provider that returns SionnaStepResult. Without a provider it uses the mock
    adapter, which lets the rest of CPLE be validated on CPU-only machines.
    """

    def __init__(self, provider: Any | None = None, fallback_config: MockAdapterConfig | None = None, tti_ms: float = 5.0):
        self.env = inspect_sionna_environment()
        self.provider = provider or MockSionnaAdapter(fallback_config or MockAdapterConfig(), tti_ms=tti_ms)

    def reset(self, seed: int | None = None) -> None:
        if hasattr(self.provider, "reset"):
            self.provider.reset(seed)

    def step(self, slot_idx: int) -> SionnaStepResult:
        result = self.provider.step(slot_idx)
        if not isinstance(result, SionnaStepResult):
            raise TypeError("SionnaAdapter provider must return SionnaStepResult")
        return result
