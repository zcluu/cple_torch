from __future__ import annotations

import torch

from .api import CPLEContext, CPLEModelAPI, CPLESerialModelAPI, CSIServiceResult
from .events import EventType
from .logger import CPLEEventLogger
from .profiler import OnlineRuntimeProfiler
from .stage import StageDAGExecutor


class ModelDispatcher:
    def __init__(self, profiler: OnlineRuntimeProfiler, logger: CPLEEventLogger, deadline_ms: float):
        self.profiler = profiler
        self.logger = logger
        self.deadline_ms = deadline_ms
        self.stage_executor = StageDAGExecutor()

    def run(self, model: CPLEModelAPI, context: CPLEContext) -> CSIServiceResult:
        if isinstance(model, CPLESerialModelAPI):
            return self._run_serial(model, context)
        return self._run_parallel(model, context)

    def _run_parallel(self, model: CPLEModelAPI, context: CPLEContext) -> CSIServiceResult:
        model_input = model.prepare_input(context)
        self.logger.record(
            slot_idx=context.slot_idx,
            sim_time_ms=context.sim_time_ms,
            event_type=EventType.MODEL_START,
            ue_id=context.ue_id,
            bs_id=context.bs_id,
            model_name=model.name,
            mode=model.mode,
            device=context.device,
            deadline_ms=self.deadline_ms,
        )
        with torch.inference_mode():
            profile = self.profiler.measure(model.forward, model_input, device=context.device)
        result = model.parse_output(profile.output, context)
        capability = model.capability()
        result.validate(capability.output_frames)
        self.logger.record(
            slot_idx=context.slot_idx,
            sim_time_ms=context.sim_time_ms,
            event_type=EventType.MODEL_END,
            ue_id=context.ue_id,
            bs_id=context.bs_id,
            model_name=model.name,
            mode=model.mode,
            runtime_ms=profile.runtime_ms,
            device=profile.device,
            operation_type="joint",
            output_frames=sorted(result.frames),
            deadline_ms=self.deadline_ms,
            metadata={"sync_mode": profile.sync_mode},
        )
        self.logger.record(
            slot_idx=context.slot_idx,
            sim_time_ms=context.sim_time_ms,
            event_type=EventType.DELIVERY,
            ue_id=context.ue_id,
            bs_id=context.bs_id,
            model_name=model.name,
            mode=model.mode,
            runtime_ms=profile.runtime_ms,
            device=profile.device,
            operation_type="joint",
            output_frames=sorted(result.frames),
            deadline_ms=self.deadline_ms,
            metadata={
                "critical_path_ms": profile.runtime_ms,
                "feedback_requests": capability.feedback_requests,
            },
        )
        return result

    def _run_serial(self, model: CPLESerialModelAPI, context: CPLEContext) -> CSIServiceResult:
        current_input = model.prepare_input(context)
        outputs = {}
        total_runtime = 0.0
        ordered_stages = self.stage_executor.order(model.stages())
        for stage in ordered_stages:
            self.logger.record(
                slot_idx=context.slot_idx,
                sim_time_ms=context.sim_time_ms,
                event_type=EventType.STAGE_START,
                ue_id=context.ue_id,
                bs_id=context.bs_id,
                model_name=model.name,
                mode=model.mode,
                stage_name=stage.name,
                device=context.device,
                deadline_ms=self.deadline_ms,
            )
            if stage.input_from is not None:
                if stage.input_from not in outputs:
                    raise ValueError(f"Stage {stage.name} requires missing input_from stage: {stage.input_from}")
                stage_input = outputs[stage.input_from]
            else:
                stage_input = current_input
            with torch.inference_mode():
                profile = self.profiler.measure(stage.fn, stage_input, device=context.device)
            current_input = profile.output
            total_runtime += profile.runtime_ms
            self.logger.record(
                slot_idx=context.slot_idx,
                sim_time_ms=context.sim_time_ms,
                event_type=EventType.STAGE_END,
                ue_id=context.ue_id,
                bs_id=context.bs_id,
                model_name=model.name,
                mode=model.mode,
                stage_name=stage.name,
                runtime_ms=profile.runtime_ms,
                device=profile.device,
                operation_type=stage.operation_type,
                output_frames=stage.output_frames,
                deadline_ms=self.deadline_ms,
                metadata={"sync_mode": profile.sync_mode},
            )
            outputs[stage.name] = current_input
        result = model.parse_output(outputs, context)
        capability = model.capability()
        result.validate(capability.output_frames)
        self.logger.record(
            slot_idx=context.slot_idx,
            sim_time_ms=context.sim_time_ms,
            event_type=EventType.DELIVERY,
            ue_id=context.ue_id,
            bs_id=context.bs_id,
            model_name=model.name,
            mode=model.mode,
            runtime_ms=total_runtime,
            device=context.device,
            operation_type="serial_pipeline",
            output_frames=sorted(result.frames),
            deadline_ms=self.deadline_ms,
            metadata={
                "critical_path_ms": total_runtime,
                "feedback_requests": capability.feedback_requests,
            },
        )
        return result
