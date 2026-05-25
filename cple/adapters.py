from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SionnaStepResult:
    slot_idx: int
    sim_time_ms: float
    scheduled_ues: list[int]
    h_t: dict[int, torch.Tensor]
    h_history: dict[int, list[torch.Tensor]]
    bs_id: int = 0
    sinr: Any = None
    bler: Any = None
    mcs: Any = None
    raw_state: Any = None
    feedback_resources_by_ue: dict[int, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
