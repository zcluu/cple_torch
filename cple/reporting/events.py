from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    SLOT_START = "slot_start"
    SLOT_END = "slot_end"
    FLOW_START = "flow_start"
    STAGE_END = "stage_end"
    FLOW_END = "flow_end"


@dataclass
class CPLEEvent:
    run_id: str
    event_idx: int
    slot_idx: int
    sim_time_ms: float
    event_type: str
    wall_time_ms: float
    ue_id: int | None = None
    bs_id: int | None = None
    flow_name: str | None = None
    flow_kind: str | None = None
    stage_name: str | None = None
    stage_kind: str | None = None
    side: str | None = None
    frame_idx: int | None = None
    runtime_ms: float | None = None
    start_ms: float | None = None
    end_ms: float | None = None
    payload_bits: int | None = None
    total_latency_ms: float | None = None
    deadline_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["metadata"] = repr(self.metadata)
        return row
