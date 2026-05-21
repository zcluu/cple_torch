import time

import torch

from cple.adapters import MockSionnaAdapter
from cple.api import CPLEStage
from cple.config import MockAdapterConfig
from cple.profiler import OnlineRuntimeProfiler
from cple.scheduler import FeedbackScheduler
from cple.stage import StageDAGExecutor


def test_profiler_cpu_sleep():
    profiler = OnlineRuntimeProfiler()
    result = profiler.measure(lambda: time.sleep(0.01), device="cpu")
    assert result.runtime_ms >= 8.0
    assert result.device == "cpu"


def test_stage_dag_validation():
    executor = StageDAGExecutor()
    stages = [
        CPLEStage("a", lambda x: x),
        CPLEStage("b", lambda x: x, depends_on=["a"]),
    ]
    assert [stage.name for stage in executor.order(stages)] == ["a", "b"]
    cyclic = [
        CPLEStage("a", lambda x: x, depends_on=["b"]),
        CPLEStage("b", lambda x: x, depends_on=["a"]),
    ]
    try:
        executor.order(cyclic)
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("cycle should fail")


def test_feedback_scheduler_capacity_creates_wait():
    scheduler = FeedbackScheduler(capacity_per_slot=2, slot_ms=1.0)
    first = scheduler.schedule(request_time_ms=0.0, feedback_requests=2)
    second = scheduler.schedule(request_time_ms=0.0, feedback_requests=1)
    assert first.scheduling_delay_ms == 0.0
    assert first.feedback_duration_ms == 1.0
    assert second.scheduling_delay_ms == 1.0
    assert second.feedback_duration_ms == 0.5


def test_mock_adapter_reproducible_shape():
    config = MockAdapterConfig(num_ues=3, csi_dim=5, scheduled_per_slot=2, seed=3)
    adapter = MockSionnaAdapter(config)
    step = adapter.step(0)
    assert step.scheduled_ues == [0, 1]
    assert isinstance(step.h_t[0], torch.Tensor)
    assert step.h_t[0].shape == (5,)
