import time

from cple.api import CPLEStage
from cple.profiler import OnlineRuntimeProfiler
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

