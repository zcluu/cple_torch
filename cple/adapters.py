from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .config import MockAdapterConfig


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
    metadata: dict[str, Any] = field(default_factory=dict)


class MockSionnaAdapter:
    def __init__(self, config: MockAdapterConfig, tti_ms: float = 5.0):
        self.config = config
        self.tti_ms = tti_ms
        self.generator = torch.Generator().manual_seed(config.seed)
        self._history: dict[int, list[torch.Tensor]] = {ue: [] for ue in range(config.num_ues)}

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)
        self._history = {ue: [] for ue in range(self.config.num_ues)}

    def step(self, slot_idx: int) -> SionnaStepResult:
        start = slot_idx % self.config.num_ues
        scheduled = [(start + i) % self.config.num_ues for i in range(self.config.scheduled_per_slot)]
        h_t: dict[int, torch.Tensor] = {}
        h_history: dict[int, list[torch.Tensor]] = {}
        for ue in range(self.config.num_ues):
            trend = torch.full((self.config.csi_dim,), float(ue) * 0.01 + slot_idx * 0.001)
            noise = torch.randn(self.config.csi_dim, generator=self.generator) * 0.01
            current = trend + noise
            self._history[ue].append(current)
            self._history[ue] = self._history[ue][-self.config.history_len :]
            h_t[ue] = current
            h_history[ue] = list(self._history[ue])
        return SionnaStepResult(
            slot_idx=slot_idx,
            sim_time_ms=slot_idx * self.tti_ms,
            scheduled_ues=scheduled,
            h_t=h_t,
            h_history=h_history,
            metadata={"adapter": "mock"},
        )
