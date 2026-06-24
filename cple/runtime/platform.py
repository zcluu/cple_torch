from __future__ import annotations

from pathlib import Path

from ..api import CPLEContext, FlowModel
from ..configs.schema import FeedbackConfig, PlatformConfig
from ..reporting.events import EventType
from ..reporting.logger import CPLEEventLogger
from ..reporting.metrics import CPLEMetricsEngine
from .executor import FlowExecutor
from .profiler import OnlineRuntimeProfiler


class CPLEPlatform:
    def __init__(self, config: PlatformConfig, shape, feedback: FeedbackConfig, adapter):
        self.config = config
        self.shape = shape
        self.feedback = feedback
        self.adapter = adapter
        self.logger = CPLEEventLogger(run_id=config.run_id)
        self.profiler = OnlineRuntimeProfiler()
        self.executor = FlowExecutor(self.profiler, adapter, feedback.to_spec())
        self.metrics = CPLEMetricsEngine()
        if not hasattr(adapter, "schedule_feedback"):
            raise TypeError("CPLEPlatform requires an adapter with schedule_feedback")

    def build_context(self, step_result, ue_id: int) -> CPLEContext:
        return CPLEContext(
            run_id=self.config.run_id,
            slot_idx=step_result.slot_idx,
            tti_ms=self.config.tti_ms,
            ue_id=ue_id,
            bs_id=step_result.bs_id,
            device=self.config.device,
            shape=self.shape,
            window=step_result.windows[ue_id],
            sionna_state=step_result,
        )

    def run(self, network) -> dict[str, object]:
        flow = network.build_flow(self.shape) if hasattr(network, "build_flow") else network
        if not isinstance(flow, FlowModel):
            raise TypeError("run expects SerialNetwork, ParallelNetwork, or FlowModel")
        self.logger = CPLEEventLogger(run_id=self.config.run_id)
        if hasattr(self.adapter, "reset"):
            self.adapter.reset(self.config.seed)
        for slot in range(self.config.num_slots):
            step_result = self.adapter.step(slot)
            self.logger.record(
                slot_idx=slot,
                sim_time_ms=step_result.sim_time_ms,
                event_type=EventType.SLOT_START,
                metadata={
                    "scheduled_ues": step_result.scheduled_ues,
                    "warmup": self._is_warmup(slot),
                },
            )
            for ue_id in step_result.scheduled_ues:
                context = self.build_context(step_result, ue_id)
                self._run_flow(flow, context)
            self.logger.record(
                slot_idx=slot,
                sim_time_ms=step_result.sim_time_ms,
                event_type=EventType.SLOT_END,
                metadata={"warmup": self._is_warmup(slot)},
            )
        return self.compute_metrics()

    def compute_metrics(self) -> dict[str, object]:
        return self.metrics.compute_all(self._metric_events())

    def export_outputs(self, output_dir: str | Path) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        self.logger.export_csv(output_dir / "event_log.csv")
        paths["event_log"] = output_dir / "event_log.csv"
        summaries = self.compute_metrics()
        for name, frame in summaries.items():
            path = output_dir / f"{name}.csv"
            frame.to_csv(path, index=False)
            paths[name] = path
        return paths

    def _run_flow(self, flow: FlowModel, context: CPLEContext) -> None:
        warmup = self._is_warmup(context.slot_idx)
        self.logger.record(
            slot_idx=context.slot_idx,
            sim_time_ms=context.sim_time_ms,
            event_type=EventType.FLOW_START,
            ue_id=context.ue_id,
            bs_id=context.bs_id,
            flow_name=flow.name,
            flow_kind=flow.flow.value,
            deadline_ms=self.config.deadline_ms,
            metadata={"warmup": warmup},
        )
        result = self.executor.run(flow, context)
        for timing in result.stage_timings:
            self.logger.record(
                slot_idx=context.slot_idx,
                sim_time_ms=context.sim_time_ms,
                event_type=EventType.STAGE_END,
                ue_id=context.ue_id,
                bs_id=context.bs_id,
                flow_name=flow.name,
                flow_kind=flow.flow.value,
                stage_name=timing.name,
                stage_kind=timing.kind.value,
                side=timing.side.value,
                frame_idx=timing.frame_idx,
                runtime_ms=timing.runtime_ms,
                start_ms=timing.start_ms,
                end_ms=timing.end_ms,
                payload_bits=timing.payload_bits,
                deadline_ms=self.config.deadline_ms,
                metadata={**timing.metadata, "warmup": warmup},
            )
        self.logger.record(
            slot_idx=context.slot_idx,
            sim_time_ms=context.sim_time_ms,
            event_type=EventType.FLOW_END,
            ue_id=context.ue_id,
            bs_id=context.bs_id,
            flow_name=flow.name,
            flow_kind=flow.flow.value,
            total_latency_ms=result.total_latency_ms,
            deadline_ms=self.config.deadline_ms,
            metadata={**result.metadata, "warmup": warmup},
        )

    def _metric_events(self):
        return [
            event
            for event in self.logger.events
            if not event.metadata.get("warmup", False)
        ]

    def _is_warmup(self, slot_idx: int) -> bool:
        return slot_idx < self.config.warmup_slots
