from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from .events import CPLEEvent, EventType


class CPLEEventLogger:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.events: list[CPLEEvent] = []
        self._event_idx = 0

    def now_ms(self) -> float:
        return time.perf_counter() * 1000.0

    def record(
        self,
        *,
        slot_idx: int,
        sim_time_ms: float,
        event_type: EventType | str,
        ue_id: int | None = None,
        bs_id: int | None = None,
        model_name: str | None = None,
        mode: str | None = None,
        stage_name: str | None = None,
        runtime_ms: float | None = None,
        scheduling_delay_ms: float | None = None,
        feedback_duration_ms: float | None = None,
        total_latency_ms: float | None = None,
        device: str | None = None,
        operation_type: str | None = None,
        output_frames: list[int] | None = None,
        deadline_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CPLEEvent:
        event = CPLEEvent(
            run_id=self.run_id,
            event_idx=self._event_idx,
            slot_idx=slot_idx,
            sim_time_ms=sim_time_ms,
            ue_id=ue_id,
            bs_id=bs_id,
            model_name=model_name,
            mode=mode,
            stage_name=stage_name,
            event_type=str(event_type),
            wall_time_ms=self.now_ms(),
            runtime_ms=runtime_ms,
            scheduling_delay_ms=scheduling_delay_ms,
            feedback_duration_ms=feedback_duration_ms,
            total_latency_ms=total_latency_ms,
            device=device,
            operation_type=operation_type,
            output_frames=output_frames or [],
            deadline_ms=deadline_ms,
            metadata=metadata or {},
        )
        self._event_idx += 1
        self.events.append(event)
        return event

    def export_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [event.to_dict() for event in self.events]
        fieldnames = list(rows[0].keys()) if rows else list(CPLEEvent.__dataclass_fields__)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
