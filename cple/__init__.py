from __future__ import annotations

from .api import (
    BSInput,
    CPLEContext,
    CSIShapeSpec,
    CSIWindow,
    ExecutionSide,
    FeedbackSpec,
    FlowKind,
    FlowModel,
    FlowResult,
    ModelStep,
    ParallelNetwork,
    SerialNetwork,
    StageKind,
    StageTiming,
)
from .data.scenario import (
    SionnaScenarioConfig,
    describe_sionna_mapping,
    load_sionna_scenario,
    scenario_to_adapter_config,
    scenario_to_feedback_config,
    scenario_to_platform_config,
    scenario_to_shape_config,
    validate_sionna_scenario,
)
from .reporting.events import CPLEEvent, EventType
from .reporting.logger import CPLEEventLogger

__all__ = [
    "BSInput",
    "CPLEContext",
    "CSIShapeSpec",
    "CSIWindow",
    "ExecutionSide",
    "FeedbackSpec",
    "FlowKind",
    "FlowModel",
    "FlowResult",
    "ModelStep",
    "ParallelNetwork",
    "SerialNetwork",
    "StageKind",
    "StageTiming",
    "CPLEEvent",
    "EventType",
    "CPLEEventLogger",
    "SionnaScenarioConfig",
    "describe_sionna_mapping",
    "load_sionna_scenario",
    "scenario_to_adapter_config",
    "scenario_to_feedback_config",
    "scenario_to_platform_config",
    "scenario_to_shape_config",
    "validate_sionna_scenario",
    "run_experiment",
    "build_adapter",
    "build_flow",
    "CPLEPlatform",
    "CPLEMetricsEngine",
    "OnlineRuntimeProfiler",
    "SionnaSysAdapter",
    "LinearBSPart",
    "LinearPredictor",
    "LinearUEPart",
    "build_dummy_flow",
]


def __getattr__(name: str):
    if name in {"run_experiment", "build_adapter", "build_flow"}:
        from .configs.runner import build_adapter, build_flow, run_experiment

        return {"run_experiment": run_experiment, "build_adapter": build_adapter, "build_flow": build_flow}[name]
    if name == "CPLEPlatform":
        from .runtime.platform import CPLEPlatform

        return CPLEPlatform
    if name == "CPLEMetricsEngine":
        from .reporting.metrics import CPLEMetricsEngine

        return CPLEMetricsEngine
    if name == "OnlineRuntimeProfiler":
        from .runtime.profiler import OnlineRuntimeProfiler

        return OnlineRuntimeProfiler
    if name == "SionnaSysAdapter":
        from .data.sionna_sys import SionnaSysAdapter

        return SionnaSysAdapter
    if name in {"LinearBSPart", "LinearPredictor", "LinearUEPart", "build_dummy_flow"}:
        from .models import LinearBSPart, LinearPredictor, LinearUEPart, build_dummy_flow

        return {
            "LinearBSPart": LinearBSPart,
            "LinearPredictor": LinearPredictor,
            "LinearUEPart": LinearUEPart,
            "build_dummy_flow": build_dummy_flow,
        }[name]
    raise AttributeError(name)
