from __future__ import annotations

from math import ceil
from typing import Any

import torch

from ..api import (
    BSInput,
    CPLEContext,
    ExecutionSide,
    FeedbackSpec,
    FlowKind,
    FlowModel,
    FlowResult,
    ModelStep,
    StageKind,
    StageTiming,
)
from .profiler import OnlineRuntimeProfiler


class FlowExecutor:
    def __init__(self, profiler: OnlineRuntimeProfiler, adapter, feedback: FeedbackSpec):
        self.profiler = profiler
        self.adapter = adapter
        self.feedback = feedback

    def run(self, flow: FlowModel, context: CPLEContext) -> FlowResult:
        feedback_frames = flow.resolved_feedback_frames(context.shape)
        ue_output, ue_stages, ue_payload_outputs = self._measure_steps(
            flow.ue_steps,
            context.window,
            context,
            StageKind.UE_MODEL,
            ExecutionSide.UE,
            start_ms=context.sim_time_ms,
            feedback_frames=feedback_frames,
        )
        ue_end_ms = ue_stages[-1].end_ms
        air_timings = self._schedule_air_feedback(
            context=context,
            schedule_request_time_ms=context.sim_time_ms,
            timeline_start_ms=ue_end_ms,
            payload_source=ue_payload_outputs,
            feedback_frames=feedback_frames,
        )
        air_end_ms = max(timing.end_ms for timing in air_timings)

        bs_input = BSInput(
            window=context.window,
            ue_output=ue_output,
            flow=flow.flow,
            feedback_frames=feedback_frames,
            metadata=flow.metadata,
        )
        _, bs_stages, _ = self._measure_steps(
            flow.bs_steps,
            bs_input,
            context,
            StageKind.BS_MODEL,
            ExecutionSide.BS,
            start_ms=air_end_ms,
            feedback_frames=feedback_frames,
        )
        total_end_ms = max(air_end_ms, bs_stages[-1].end_ms)
        result = FlowResult(
            flow=flow.flow,
            total_latency_ms=total_end_ms - context.sim_time_ms,
            stage_timings=[*ue_stages, *air_timings, *bs_stages],
            metadata={
                "feedback_frames": feedback_frames,
                "bs_start_ms": air_end_ms,
                "tool_scope": "model_steps_air_feedback",
                **flow.metadata,
            },
        )
        result.validate()
        return result

    def _measure_steps(
        self,
        steps: list[ModelStep],
        initial_arg: Any,
        context: CPLEContext,
        kind: StageKind,
        side: ExecutionSide,
        *,
        start_ms: float,
        feedback_frames: int,
    ) -> tuple[Any, list[StageTiming], list[Any]]:
        if not steps:
            raise ValueError("flow must provide at least one model step per side")
        output = initial_arg
        timings: list[StageTiming] = []
        final_step_outputs: list[Any] = []
        cursor_ms = start_ms
        for step in steps:
            step_input = output
            step_outputs: list[Any] = []
            repeat_count = step.resolved_repeat(feedback_frames)
            for repeat_idx in range(repeat_count):
                measured_input = self._step_input(
                    step,
                    step_input,
                    context,
                    repeat_idx=repeat_idx,
                    feedback_frames=feedback_frames,
                )
                output, timing = self._measure(
                    step.module,
                    measured_input,
                    context,
                    kind,
                    side,
                    start_ms=cursor_ms,
                    name=step.name,
                    repeat_idx=repeat_idx,
                )
                step_outputs.append(output)
                timings.append(timing)
                cursor_ms = timing.end_ms
            final_step_outputs = step_outputs
            if repeat_count > 1 and step.input_source == "pred_fb_frame":
                output = step_outputs
        return output, timings, final_step_outputs

    def _step_input(
        self,
        step: ModelStep,
        previous_output: Any,
        context: CPLEContext,
        *,
        repeat_idx: int,
        feedback_frames: int,
    ) -> Any:
        if step.input_source == "previous":
            return previous_output
        if step.input_source == "pred_fb_frame":
            return self._pred_fb_frame_input(previous_output, context, repeat_idx, feedback_frames)
        raise ValueError(f"unsupported ModelStep.input_source: {step.input_source}")

    def _pred_fb_frame_input(
        self,
        predicted_future: Any,
        context: CPLEContext,
        repeat_idx: int,
        feedback_frames: int,
    ) -> torch.Tensor:
        result_frames = self._pred_fb_result_frames(predicted_future, context)
        if feedback_frames != result_frames.shape[0]:
            raise ValueError(
                f"pred-fb feedback_frames={feedback_frames} does not match result frames={result_frames.shape[0]}"
            )
        return result_frames[repeat_idx]

    def _pred_fb_result_frames(self, predicted_future: Any, context: CPLEContext) -> torch.Tensor:
        if not isinstance(predicted_future, torch.Tensor):
            raise TypeError("pred-fb frame input requires predictor output to be a torch.Tensor")
        future = predicted_future
        frame_shape = context.shape.frame_shape
        if tuple(future.shape) == frame_shape:
            future = future.reshape(1, *frame_shape)
        expected_future_shape = (context.shape.horizon, *frame_shape)
        if tuple(future.shape) == expected_future_shape:
            current = context.window.current.to(device=future.device, dtype=future.dtype).reshape(1, *frame_shape)
            return torch.cat([current, future], dim=0)
        expected_result_shape = (context.shape.output_frames, *frame_shape)
        if tuple(future.shape) == expected_result_shape:
            return future
        raise ValueError(
            f"pred-fb predictor output shape {tuple(predicted_future.shape)} must be "
            f"{expected_future_shape} or {expected_result_shape}"
        )

    def _measure(
        self,
        fn,
        arg: Any,
        context: CPLEContext,
        kind: StageKind,
        side: ExecutionSide,
        *,
        start_ms: float,
        name: str | None = None,
        repeat_idx: int | None = None,
    ) -> tuple[Any, StageTiming]:
        with torch.inference_mode():
            profile = self.profiler.measure(fn, arg, device=context.device)
        return profile.output, StageTiming(
            name=name or getattr(fn, "name", kind.value),
            kind=kind,
            side=side,
            runtime_ms=profile.runtime_ms,
            start_ms=start_ms,
            end_ms=start_ms + profile.runtime_ms,
            metadata={
                "sync_mode": profile.sync_mode,
                "device": profile.device,
                "repeat_idx": repeat_idx,
            },
        )

    def _schedule_air_feedback(
        self,
        *,
        context: CPLEContext,
        schedule_request_time_ms: float,
        timeline_start_ms: float,
        payload_source: Any,
        feedback_frames: int,
    ) -> list[StageTiming]:
        payload_bits = self._payload_bits(payload_source, context, feedback_frames)
        bits_per_frame = ceil(payload_bits / feedback_frames)
        timings: list[StageTiming] = []
        schedule_cursor_ms = schedule_request_time_ms
        timeline_cursor_ms = timeline_start_ms
        for frame_idx in range(feedback_frames):
            schedule = self.adapter.schedule_feedback(
                ue_id=context.ue_id,
                request_time_ms=schedule_cursor_ms,
                payload_bits=bits_per_frame,
            )
            runtime_ms = schedule.scheduling_delay_ms + schedule.feedback_duration_ms
            timings.append(
                StageTiming(
                    name="air_feedback",
                    kind=StageKind.AIR_FEEDBACK,
                    side=ExecutionSide.AIR,
                    runtime_ms=runtime_ms,
                    start_ms=timeline_cursor_ms,
                    end_ms=timeline_cursor_ms + runtime_ms,
                    frame_idx=frame_idx,
                    payload_bits=bits_per_frame,
                    metadata={
                        "schedule_request_time_ms": schedule.request_time_ms,
                        "schedule_start_time_ms": schedule.start_time_ms,
                        "schedule_finish_time_ms": schedule.finish_time_ms,
                        "scheduling_delay_ms": schedule.scheduling_delay_ms,
                        "feedback_duration_ms": schedule.feedback_duration_ms,
                        "resource_units": schedule.resource_units,
                        "resource_units_used": schedule.resource_units_used,
                    },
                )
            )
            schedule_cursor_ms = schedule.finish_time_ms
            timeline_cursor_ms += runtime_ms
        return timings

    def _payload_bits(self, ue_output: Any, context: CPLEContext, feedback_frames: int) -> int:
        explicit_payload_bits = self._explicit_payload_bits(ue_output)
        if explicit_payload_bits is not None:
            return max(1, ceil(explicit_payload_bits))
        if isinstance(ue_output, (list, tuple)):
            return max(1, sum(self._payload_bits(item, context, 1) for item in ue_output))
        if isinstance(ue_output, torch.Tensor):
            elements = ue_output.numel()
            if torch.is_complex(ue_output):
                elements *= 2
        else:
            elements = context.shape.elements_per_frame * feedback_frames
        return max(1, ceil(elements * self.feedback.bitwidth))

    def _explicit_payload_bits(self, ue_output: Any) -> int | float | None:
        if isinstance(ue_output, dict) and "payload_bits" in ue_output:
            return ue_output["payload_bits"]
        return getattr(ue_output, "payload_bits", None)
