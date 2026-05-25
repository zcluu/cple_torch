from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedbackScheduleResult:
    request_time_ms: float
    start_time_ms: float
    finish_time_ms: float
    scheduling_delay_ms: float
    feedback_duration_ms: float
