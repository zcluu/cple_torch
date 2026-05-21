from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal


@dataclass
class CSIServiceResult:
    frames: dict[int, Any]
    valid_from_slot: int
    valid_until_slot: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, allowed_frames: list[int] | None = None) -> None:
        if not self.frames:
            raise ValueError("CSIServiceResult.frames must not be empty")
        if 0 not in self.frames:
            raise ValueError("CSIServiceResult must include frame 0")
        if self.valid_until_slot < self.valid_from_slot:
            raise ValueError("valid_until_slot must be >= valid_from_slot")
        if allowed_frames is not None:
            unexpected = set(self.frames) - set(allowed_frames)
            if unexpected:
                raise ValueError(f"Unexpected output frames: {sorted(unexpected)}")


@dataclass
class CPLEContext:
    run_id: str
    slot_idx: int
    tti_ms: float
    ue_id: int
    bs_id: int
    h_t: Any
    h_history: Any
    scheduled: bool
    device: str
    sionna_state: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sim_time_ms(self) -> float:
        return self.slot_idx * self.tti_ms


@dataclass
class ModelCapability:
    output_frames: list[int]
    requires_history: int = 0
    supports_batch_ue: bool = False
    device: str = "cpu"
    feedback_requests: int = 1


@dataclass
class CPLEStage:
    name: str
    fn: Callable[[Any], Any]
    depends_on: list[str] = field(default_factory=list)
    input_from: str | None = None
    output_frames: list[int] = field(default_factory=list)
    operation_type: Literal["prediction", "feedback", "joint", "other"] = "other"


class CPLEModelAPI(ABC):
    name: str
    mode: Literal["serial", "parallel"]

    def reset(self, ue_id: int) -> None:
        return None

    @abstractmethod
    def prepare_input(self, context: CPLEContext) -> Any:
        raise NotImplementedError

    @abstractmethod
    def forward(self, model_input: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def parse_output(self, output: Any, context: CPLEContext) -> CSIServiceResult:
        raise NotImplementedError

    @abstractmethod
    def capability(self) -> ModelCapability:
        raise NotImplementedError


class CPLEParallelModelAPI(CPLEModelAPI):
    mode: Literal["parallel"] = "parallel"

    def output_frames(self) -> list[int]:
        return self.capability().output_frames


class CPLESerialModelAPI(CPLEModelAPI):
    mode: Literal["serial"] = "serial"

    @abstractmethod
    def stages(self) -> list[CPLEStage]:
        raise NotImplementedError
