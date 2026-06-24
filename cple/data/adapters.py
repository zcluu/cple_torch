from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..api import CSIWindow


@dataclass
class SionnaStepResult:
    slot_idx: int
    sim_time_ms: float
    scheduled_ues: list[int]
    windows: dict[int, CSIWindow]
    bs_id: int = 0
    raw_state: Any = None
    feedback_resources_by_ue: dict[int, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeedbackScheduleResult:
    request_time_ms: float
    start_time_ms: float
    finish_time_ms: float
    scheduling_delay_ms: float
    feedback_duration_ms: float
    payload_bits: int
    resource_units: int
    resource_units_used: int
    metadata: dict[str, Any] = field(default_factory=dict)
