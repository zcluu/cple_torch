from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from ..api import CSIShapeSpec, FeedbackSpec, FlowKind


@dataclass
class PlatformConfig:
    run_id: str = "smoke"
    num_slots: int = 20
    tti_ms: float = 1.0
    deadline_ms: float = 5.0
    device: str = "cpu"
    warmup_slots: int = 1
    output_dir: str = "outputs/smoke"
    seed: int = 7

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("platform.run_id must not be empty")
        if self.num_slots <= 0:
            raise ValueError("platform.num_slots must be positive")
        if self.tti_ms <= 0:
            raise ValueError("platform.tti_ms must be positive")
        if self.deadline_ms <= 0:
            raise ValueError("platform.deadline_ms must be positive")
        if self.warmup_slots < 0:
            raise ValueError("platform.warmup_slots must be non-negative")
        if self.warmup_slots >= self.num_slots:
            raise ValueError("platform.warmup_slots must be smaller than num_slots")
        if not self.device:
            raise ValueError("platform.device must not be empty")
        if not self.output_dir:
            raise ValueError("platform.output_dir must not be empty")


@dataclass
class ShapeConfig:
    frame_shape: tuple[int, ...] = (2, 4, 4)
    axes: tuple[str, ...] = ("complex", "rx_ant", "subcarrier")
    history_len: int = 4
    horizon: int = 3
    dtype: str = "float32"

    def to_spec(self) -> CSIShapeSpec:
        return CSIShapeSpec(
            frame_shape=tuple(self.frame_shape),
            axes=tuple(self.axes),
            history_len=self.history_len,
            horizon=self.horizon,
            dtype=self.dtype,
        )

    def validate(self) -> None:
        self.to_spec()


@dataclass
class FeedbackConfig:
    bitwidth: int = 8
    bits_per_resource_unit: int = 64

    def to_spec(self) -> FeedbackSpec:
        return FeedbackSpec(
            bitwidth=self.bitwidth,
            bits_per_resource_unit=self.bits_per_resource_unit,
        )

    def validate(self) -> None:
        self.to_spec()


@dataclass
class AdapterConfig:
    num_ues: int = 4
    scheduled_per_slot: int = 2
    seed: int = 7

    def validate(self) -> None:
        if self.num_ues <= 0:
            raise ValueError("adapter.num_ues must be positive")
        if self.scheduled_per_slot <= 0:
            raise ValueError("adapter.scheduled_per_slot must be positive")
        if self.scheduled_per_slot > self.num_ues:
            raise ValueError("adapter.scheduled_per_slot cannot exceed num_ues")


@dataclass
class ExperimentConfig:
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    flow: str = FlowKind.FB_PRED.value
    sionna_scenario_path: str | None = None

    def validate(self) -> None:
        self.platform.validate()
        self.shape.validate()
        self.feedback.validate()
        self.adapter.validate()
        FlowKind(self.flow)


def merge_dataclass(cls_or_obj, data: dict[str, Any] | None):
    obj = cls_or_obj() if callable(cls_or_obj) else cls_or_obj
    for key, value in (data or {}).items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown {type(obj).__name__} field: {key}")
        current = getattr(obj, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        elif isinstance(current, float) and isinstance(value, str):
            value = float(value)
        elif isinstance(current, int) and isinstance(value, str):
            value = int(value)
        setattr(obj, key, value)
    return obj


def dataclass_field_names(cls) -> set[str]:
    return {field.name for field in fields(cls)}
