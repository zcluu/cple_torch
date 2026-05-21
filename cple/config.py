from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlatformConfig:
    run_id: str = "smoke"
    num_slots: int = 20
    tti_ms: float = 5.0
    deadline_ms: float = 5.0
    device: str = "cpu"
    feedback_capacity_per_slot: int = 4
    feedback_slot_ms: float | None = None
    warmup_slots: int = 1
    output_dir: str = "outputs/smoke"
    seed: int = 7
    num_workers: int = 1


@dataclass
class MockAdapterConfig:
    num_ues: int = 4
    csi_dim: int = 8
    scheduled_per_slot: int = 2
    history_len: int = 2
    seed: int = 7


@dataclass
class ExperimentConfig:
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    adapter: MockAdapterConfig = field(default_factory=MockAdapterConfig)
    models: list[str] = field(default_factory=lambda: ["dummy_parallel", "dummy_serial"])
    sionna_scenario_path: str | None = None
