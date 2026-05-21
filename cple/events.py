from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    SLOT_START = "slot_start"
    SLOT_END = "slot_end"
    CSI_REQUEST = "csi_request"
    MODEL_START = "model_start"
    MODEL_END = "model_end"
    STAGE_START = "stage_start"
    STAGE_END = "stage_end"
    DELIVERY = "delivery"
    EXCEPTION = "exception"


@dataclass
class CPLEEvent:
    run_id: str
    event_idx: int
    slot_idx: int
    sim_time_ms: float
    ue_id: int | None
    bs_id: int | None
    model_name: str | None
    mode: str | None
    stage_name: str | None
    event_type: str
    wall_time_ms: float
    runtime_ms: float | None = None
    scheduling_delay_ms: float | None = None
    feedback_duration_ms: float | None = None
    total_latency_ms: float | None = None
    device: str | None = None
    operation_type: str | None = None
    output_frames: list[int] = field(default_factory=list)
    deadline_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["output_frames"] = ",".join(str(frame) for frame in self.output_frames)
        row["metadata"] = repr(self.metadata)
        return row
