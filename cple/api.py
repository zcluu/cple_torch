from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import prod
from typing import Any, Callable, Literal

import torch


class FlowKind(StrEnum):
    FB_PRED = "fb-pred"
    PRED_FB = "pred-fb"
    PARALLEL = "parallel"


class StageKind(StrEnum):
    UE_MODEL = "ue_model"
    AIR_FEEDBACK = "air_feedback"
    BS_MODEL = "bs_model"


class ExecutionSide(StrEnum):
    UE = "ue"
    BS = "bs"
    AIR = "air"


@dataclass(frozen=True)
class CSIShapeSpec:
    frame_shape: tuple[int, ...]
    axes: tuple[str, ...] = ()
    history_len: int = 5
    horizon: int = 3
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if not self.frame_shape:
            raise ValueError("frame_shape must not be empty")
        if any(dim <= 0 for dim in self.frame_shape):
            raise ValueError("frame_shape dimensions must be positive")
        if self.axes and len(self.axes) != len(self.frame_shape):
            raise ValueError("axes and frame_shape must have the same length")
        if self.axes and len(set(self.axes)) != len(self.axes):
            raise ValueError("axes must be unique")
        if self.history_len <= 0:
            raise ValueError("history_len must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.dtype not in {"float32", "complex64"}:
            raise ValueError("dtype must be float32 or complex64")

    @property
    def output_frames(self) -> int:
        return self.horizon + 1

    @property
    def elements_per_frame(self) -> int:
        return int(prod(self.frame_shape))


@dataclass(frozen=True)
class FeedbackSpec:
    bitwidth: int = 8
    bits_per_resource_unit: int = 64

    def __post_init__(self) -> None:
        if self.bitwidth <= 0:
            raise ValueError("bitwidth must be positive")
        if self.bits_per_resource_unit <= 0:
            raise ValueError("bits_per_resource_unit must be positive")


@dataclass
class CSIWindow:
    history: torch.Tensor
    current: torch.Tensor
    target: torch.Tensor
    raw: dict[str, Any] = field(default_factory=dict)

    def validate(self, shape: CSIShapeSpec) -> None:
        expected_history = (shape.history_len, *shape.frame_shape)
        expected_target = (shape.output_frames, *shape.frame_shape)
        if tuple(self.history.shape) != expected_history:
            raise ValueError(f"history shape {tuple(self.history.shape)} != {expected_history}")
        if tuple(self.current.shape) != shape.frame_shape:
            raise ValueError(f"current shape {tuple(self.current.shape)} != {shape.frame_shape}")
        if tuple(self.target.shape) != expected_target:
            raise ValueError(f"target shape {tuple(self.target.shape)} != {expected_target}")


@dataclass
class CPLEContext:
    run_id: str
    slot_idx: int
    tti_ms: float
    ue_id: int
    bs_id: int
    device: str
    shape: CSIShapeSpec
    window: CSIWindow
    sionna_state: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sim_time_ms(self) -> float:
        return self.slot_idx * self.tti_ms


@dataclass
class BSInput:
    window: CSIWindow
    ue_output: Any
    flow: FlowKind
    feedback_frames: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelStep:
    name: str
    module: Callable[[Any], Any]
    repeat: int | Literal["feedback_frames"] = 1
    input_source: Literal["previous", "pred_fb_frame"] = "previous"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ModelStep.name must not be empty")
        if not callable(self.module):
            raise TypeError(f"ModelStep.module for {self.name} must be callable")
        if self.repeat != "feedback_frames" and (not isinstance(self.repeat, int) or self.repeat <= 0):
            raise ValueError("ModelStep.repeat must be a positive integer or 'feedback_frames'")
        if self.input_source not in {"previous", "pred_fb_frame"}:
            raise ValueError("ModelStep.input_source must be previous or pred_fb_frame")

    def resolved_repeat(self, feedback_frames: int) -> int:
        repeat = feedback_frames if self.repeat == "feedback_frames" else self.repeat
        if repeat <= 0:
            raise ValueError("ModelStep.repeat must be positive")
        return repeat


@dataclass
class FlowModel:
    name: str
    flow: FlowKind
    ue_steps: list[ModelStep]
    bs_steps: list[ModelStep]
    feedback_frames: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FlowModel.name must not be empty")
        self.flow = FlowKind(self.flow)
        if not self.ue_steps:
            raise ValueError("FlowModel.ue_steps must not be empty")
        if not self.bs_steps:
            raise ValueError("FlowModel.bs_steps must not be empty")
        if self.feedback_frames is not None and self.feedback_frames <= 0:
            raise ValueError("feedback_frames must be positive")

    def resolved_feedback_frames(self, shape: CSIShapeSpec) -> int:
        if self.feedback_frames is not None:
            if self.feedback_frames <= 0:
                raise ValueError("feedback_frames must be positive")
            return self.feedback_frames
        if self.flow == FlowKind.PRED_FB:
            return shape.output_frames
        return 1


@dataclass
class SerialNetwork:
    """User-facing API for fb-pred and pred-fb latency tests."""

    name: str
    strategy: Literal["fb-pred", "pred-fb"]
    ue_steps: list[ModelStep]
    bs_steps: list[ModelStep]
    feedback_frames: int | None = None
    prediction_frames: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SerialNetwork.name must not be empty")
        if FlowKind(self.strategy) not in {FlowKind.FB_PRED, FlowKind.PRED_FB}:
            raise ValueError("SerialNetwork.strategy must be fb-pred or pred-fb")
        if not self.ue_steps:
            raise ValueError("SerialNetwork.ue_steps must not be empty")
        if not self.bs_steps:
            raise ValueError("SerialNetwork.bs_steps must not be empty")
        if self.feedback_frames is not None and self.feedback_frames <= 0:
            raise ValueError("feedback_frames must be positive")
        if self.prediction_frames is not None and self.prediction_frames <= 0:
            raise ValueError("prediction_frames must be positive")

    @classmethod
    def fb_pred(
        cls,
        *,
        name: str,
        encoder: Callable[[Any], Any],
        bs_steps: list[ModelStep] | list[tuple[str, Callable[[Any], Any]]],
        feedback_frames: int = 1,
        prediction_frames: int | None = None,
    ) -> "SerialNetwork":
        return cls(
            name=name,
            strategy=FlowKind.FB_PRED.value,
            ue_steps=[ModelStep("encoder", encoder)],
            bs_steps=_normalize_steps(bs_steps),
            feedback_frames=feedback_frames,
            prediction_frames=prediction_frames,
        )

    @classmethod
    def pred_fb(
        cls,
        *,
        name: str,
        predictor: Callable[[Any], Any] | None = None,
        encoder: Callable[[Any], Any] | None = None,
        ue_steps: list[ModelStep] | list[tuple[str, Callable[[Any], Any]]] | None = None,
        bs_steps: list[ModelStep] | list[tuple[str, Callable[[Any], Any]]],
        feedback_frames: int | None = None,
        prediction_frames: int | None = None,
    ) -> "SerialNetwork":
        if ue_steps is None:
            if predictor is None or encoder is None:
                raise ValueError("pred_fb requires either ue_steps or predictor+encoder")
            ue_steps = [
                ModelStep("predictor", predictor),
                ModelStep("encoder", encoder, repeat="feedback_frames", input_source="pred_fb_frame"),
            ]
        return cls(
            name=name,
            strategy=FlowKind.PRED_FB.value,
            ue_steps=_normalize_steps(ue_steps),
            bs_steps=_normalize_steps(bs_steps),
            feedback_frames=feedback_frames,
            prediction_frames=prediction_frames,
        )

    def build_flow(self, shape: CSIShapeSpec) -> FlowModel:
        flow = FlowKind(self.strategy)
        prediction_frames = self.prediction_frames if self.prediction_frames is not None else shape.horizon
        if prediction_frames != shape.horizon:
            raise ValueError("prediction_frames must match shape.horizon; CSIWindow target shape is derived from shape.horizon")
        frames = self.feedback_frames
        if frames is None:
            frames = prediction_frames + 1 if flow == FlowKind.PRED_FB else 1
        return FlowModel(
            name=self.name,
            flow=flow,
            ue_steps=self.ue_steps,
            bs_steps=self.bs_steps,
            feedback_frames=frames,
            metadata={"api": "serial", "prediction_frames": prediction_frames},
        )

    def run(self, platform) -> dict[str, object]:
        return platform.run(self)


@dataclass
class ParallelNetwork:
    """User-facing API for UE encoder plus BS-side whole-network latency tests."""

    name: str
    encoder: Callable[[Any], Any]
    bs_network: Callable[[Any], Any]
    feedback_frames: int = 1
    prediction_frames: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ParallelNetwork.name must not be empty")
        if not callable(self.encoder):
            raise TypeError("ParallelNetwork.encoder must be callable")
        if not callable(self.bs_network):
            raise TypeError("ParallelNetwork.bs_network must be callable")
        if self.feedback_frames <= 0:
            raise ValueError("feedback_frames must be positive")
        if self.prediction_frames is not None and self.prediction_frames <= 0:
            raise ValueError("prediction_frames must be positive")

    def build_flow(self, shape: CSIShapeSpec) -> FlowModel:
        prediction_frames = self.prediction_frames if self.prediction_frames is not None else shape.horizon
        if prediction_frames != shape.horizon:
            raise ValueError("prediction_frames must match shape.horizon; CSIWindow target shape is derived from shape.horizon")
        return FlowModel(
            name=self.name,
            flow=FlowKind.PARALLEL,
            ue_steps=[ModelStep("encoder", self.encoder)],
            bs_steps=[ModelStep("bs_network", self.bs_network)],
            feedback_frames=self.feedback_frames,
            metadata={"api": "parallel", "prediction_frames": prediction_frames},
        )

    def run(self, platform) -> dict[str, object]:
        return platform.run(self)


def _normalize_steps(
    steps: list[ModelStep] | list[tuple[str, Callable[[Any], Any]]],
) -> list[ModelStep]:
    return [step if isinstance(step, ModelStep) else ModelStep(step[0], step[1]) for step in steps]


@dataclass(frozen=True)
class StageTiming:
    name: str
    kind: StageKind
    side: ExecutionSide
    runtime_ms: float
    start_ms: float
    end_ms: float
    frame_idx: int | None = None
    payload_bits: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowResult:
    flow: FlowKind
    total_latency_ms: float
    stage_timings: list[StageTiming]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.total_latency_ms < 0:
            raise ValueError("total_latency_ms must be non-negative")
