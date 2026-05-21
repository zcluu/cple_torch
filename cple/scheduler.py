from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedbackScheduleResult:
    request_time_ms: float
    start_time_ms: float
    finish_time_ms: float
    scheduling_delay_ms: float
    feedback_duration_ms: float


class FeedbackScheduler:
    def __init__(self, capacity_per_slot: int, slot_ms: float):
        if capacity_per_slot <= 0:
            raise ValueError("capacity_per_slot must be positive")
        if slot_ms <= 0:
            raise ValueError("slot_ms must be positive")
        self.capacity_per_slot = capacity_per_slot
        self.slot_ms = slot_ms
        self._next_resource = 0

    def schedule(self, request_time_ms: float, feedback_requests: int) -> FeedbackScheduleResult:
        if feedback_requests <= 0:
            raise ValueError("feedback_requests must be positive")
        requested_resource = self._time_to_resource(request_time_ms)
        start_resource = max(self._next_resource, requested_resource)
        finish_resource = start_resource + feedback_requests
        self._next_resource = finish_resource

        start_time_ms = self._resource_to_time(start_resource)
        finish_time_ms = self._resource_to_time(finish_resource)
        return FeedbackScheduleResult(
            request_time_ms=request_time_ms,
            start_time_ms=start_time_ms,
            finish_time_ms=finish_time_ms,
            scheduling_delay_ms=max(0.0, start_time_ms - request_time_ms),
            feedback_duration_ms=finish_time_ms - start_time_ms,
        )

    def _time_to_resource(self, time_ms: float) -> int:
        slot_idx = int(time_ms // self.slot_ms)
        return slot_idx * self.capacity_per_slot

    def _resource_to_time(self, resource_idx: int) -> float:
        return (resource_idx / self.capacity_per_slot) * self.slot_ms
