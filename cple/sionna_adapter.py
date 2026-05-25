from __future__ import annotations

from dataclasses import dataclass

import torch


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
