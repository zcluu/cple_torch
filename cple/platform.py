from __future__ import annotations

from pathlib import Path

from .api import CPLEContext, CPLEModelAPI
from .config import PlatformConfig
from .dispatcher import ModelDispatcher
from .events import EventType
from .logger import CPLEEventLogger
from .metrics import CPLEMetricsEngine
from .profiler import OnlineRuntimeProfiler


class CPLEPlatform:
    def __init__(self, config: PlatformConfig, adapter, models: list[CPLEModelAPI]):
        self.config = config
        self.adapter = adapter
        self.models = models
        self.logger = CPLEEventLogger(run_id=config.run_id)
        self.profiler = OnlineRuntimeProfiler()
        self.dispatcher = ModelDispatcher(self.profiler, self.logger, deadline_ms=config.deadline_ms)
        self.metrics = CPLEMetricsEngine()
        if not hasattr(adapter, "schedule_feedback"):
            raise TypeError("CPLEPlatform requires an adapter that schedules feedback through the Sionna SYS flow")

    def build_context(self, step_result, ue_id: int) -> CPLEContext:
        return CPLEContext(
            run_id=self.config.run_id,
            slot_idx=step_result.slot_idx,
            tti_ms=self.config.tti_ms,
            ue_id=ue_id,
            bs_id=step_result.bs_id,
            h_t=step_result.h_t[ue_id],
            h_history=step_result.h_history.get(ue_id, []),
            scheduled=True,
            device=self.config.device,
            sionna_state=step_result,
        )

    def run(self) -> dict[str, object]:
        for slot in range(self.config.num_slots):
            step_result = self.adapter.step(slot)
            self.logger.record(
                slot_idx=slot,
                sim_time_ms=step_result.sim_time_ms,
                event_type=EventType.SLOT_START,
                metadata={"scheduled_ues": step_result.scheduled_ues},
            )
            for ue_id in step_result.scheduled_ues:
                context = self.build_context(step_result, ue_id)
                for model in self.models:
                    self.logger.record(
                        slot_idx=slot,
                        sim_time_ms=context.sim_time_ms,
                        event_type=EventType.CSI_REQUEST,
                        ue_id=ue_id,
                        bs_id=context.bs_id,
                        model_name=model.name,
                        mode=model.mode,
                        deadline_ms=self.config.deadline_ms,
                    )
                    self.dispatcher.run(model, context)
                    delivery = self.logger.events[-1]
                    model_runtime_ms = delivery.runtime_ms or 0.0
                    schedule = self.adapter.schedule_feedback(
                        model_name=model.name,
                        ue_id=ue_id,
                        request_time_ms=context.sim_time_ms + model_runtime_ms,
                        feedback_requests=model.capability().feedback_requests,
                        resource_units_per_request=self.config.feedback_resource_units_per_request,
                    )
                    delivery.scheduling_delay_ms = schedule.scheduling_delay_ms
                    delivery.feedback_duration_ms = schedule.feedback_duration_ms
                    delivery.total_latency_ms = (
                        model_runtime_ms
                        + schedule.scheduling_delay_ms
                        + schedule.feedback_duration_ms
                    )
            self.logger.record(
                slot_idx=slot,
                sim_time_ms=step_result.sim_time_ms,
                event_type=EventType.SLOT_END,
            )
        return self.compute_metrics()

    def compute_metrics(self) -> dict[str, object]:
        return self.metrics.compute_all(self.logger.events)

    def export_outputs(self, output_dir: str | Path) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summaries = self.compute_metrics()
        paths = {}
        for name, frame in summaries.items():
            path = output_dir / f"{name}.csv"
            frame.to_csv(path, index=False)
            paths[name] = path
        return paths
