from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PlatformConfig:
    run_id: str = "smoke"
    num_slots: int = 20
    tti_ms: float = 5.0
    deadline_ms: float = 5.0
    device: str = "cpu"
    feedback_resource_units_per_request: int = 64
    warmup_slots: int = 1
    output_dir: str = "outputs/smoke"
    seed: int = 7
    num_workers: int = 1


@dataclass
class AdapterConfig:
    num_ues: int = 4
    csi_dim: int = 8
    scheduled_per_slot: int = 2
    history_len: int = 2
    seed: int = 7


# Backward-compatible name for tests or local utilities that still construct
# the synthetic adapter directly. The official experiment path uses Sionna SYS.
MockAdapterConfig = AdapterConfig


@dataclass
class ExperimentConfig:
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    models: list[str] = field(default_factory=lambda: ["dummy_parallel", "dummy_serial"])
    adapter_type: Literal["sionna_sys"] = "sionna_sys"
    sionna_scenario_path: str | None = None
