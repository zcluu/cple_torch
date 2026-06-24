import math

import pytest
import torch
from torch import nn

from cple.data.adapters import FeedbackScheduleResult
from cple.api import BSInput, CPLEContext, CSIShapeSpec, CSIWindow, ExecutionSide, FeedbackSpec, FlowKind, ModelStep, ParallelNetwork, SerialNetwork, StageKind
from cple.models import build_lstm_mlp_network
from cple.runtime.executor import FlowExecutor
from cple.runtime.profiler import ProfileResult


class FakeProfiler:
    def __init__(self, runtimes):
        self.runtimes = iter(runtimes)

    def measure(self, fn, *args, device="cpu", **kwargs):
        runtime = next(self.runtimes)
        output = fn(*args, **kwargs)
        return ProfileResult(output=output, runtime_ms=runtime, device=device, sync_mode="fake")


class FakeAdapter:
    def __init__(self):
        self.calls = []

    def schedule_feedback(self, *, ue_id: int, request_time_ms: float, payload_bits: int):
        self.calls.append({"ue_id": ue_id, "request_time_ms": request_time_ms, "payload_bits": payload_bits})
        resource_units = math.ceil(payload_bits / 64)
        start = request_time_ms + 0.25
        finish = start + 1.0
        return FeedbackScheduleResult(
            request_time_ms=request_time_ms,
            start_time_ms=start,
            finish_time_ms=finish,
            scheduling_delay_ms=0.25,
            feedback_duration_ms=1.0,
            payload_bits=payload_bits,
            resource_units=resource_units,
            resource_units_used=resource_units,
        )


class CountingStep(nn.Module):
    name = "counting_ue"

    def __init__(self, out_features: int):
        super().__init__()
        self.out_features = out_features
        self.calls = 0

    def forward(self, data) -> torch.Tensor:
        self.calls += 1
        return torch.ones(self.out_features)


class FuturePredictor(nn.Module):
    name = "future_predictor"

    def __init__(self, shape: CSIShapeSpec):
        super().__init__()
        self.shape = shape
        self.calls = 0

    def forward(self, window: CSIWindow) -> torch.Tensor:
        self.calls += 1
        return torch.stack(
            [torch.full(self.shape.frame_shape, float(idx + 2)) for idx in range(self.shape.horizon)],
            dim=0,
        )


class RecordingEncoder(nn.Module):
    name = "recording_encoder"

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        self.inputs.append(frame.detach().clone())
        return torch.ones(8) * float(len(self.inputs))


class CountingBS(nn.Module):
    name = "counting_bs"

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, data: BSInput) -> torch.Tensor:
        self.calls.append((data.flow, data.feedback_frames, type(data.ue_output).__name__))
        return torch.zeros_like(data.window.target)


def test_shape_and_window_validation():
    shape = make_shape()
    make_context(shape).window.validate(shape)
    with pytest.raises(ValueError):
        CSIShapeSpec(frame_shape=(2, 4), axes=("complex",), history_len=4, horizon=3)


def test_fbpred_feedbacks_one_frame_then_runs_bs():
    shape = make_shape()
    context = make_context(shape)
    ue = CountingStep(out_features=8)
    bs = CountingBS()
    network = SerialNetwork.fb_pred(name="fb-pred-unit", encoder=ue, bs_steps=[("bs_network", bs)], feedback_frames=1)
    flow = network.build_flow(shape)
    adapter = FakeAdapter()
    result = FlowExecutor(FakeProfiler([0.2, 1.0]), adapter, FeedbackSpec(bitwidth=8)).run(flow, context)

    assert ue.calls == 1
    assert bs.calls == [(FlowKind.FB_PRED, 1, "Tensor")]
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["payload_bits"] == 64
    assert result.total_latency_ms == pytest.approx(2.45)


def test_predfb_feedbacks_all_output_frames_serially():
    shape = make_shape()
    context = make_context(shape)
    predictor = FuturePredictor(shape)
    encoder = RecordingEncoder()
    bs = CountingBS()
    network = SerialNetwork.pred_fb(
        name="pred-fb-unit",
        predictor=predictor,
        encoder=encoder,
        bs_steps=[("bs_network", bs)],
    )
    flow = network.build_flow(shape)
    adapter = FakeAdapter()
    result = FlowExecutor(FakeProfiler([0.2, 0.1, 0.1, 0.1, 0.1, 1.0]), adapter, FeedbackSpec(bitwidth=8)).run(flow, context)

    assert predictor.calls == 1
    assert len(encoder.inputs) == shape.output_frames
    assert torch.equal(encoder.inputs[0], context.window.current)
    assert [float(frame.flatten()[0]) for frame in encoder.inputs[1:]] == [2.0, 3.0, 4.0]
    assert bs.calls == [(FlowKind.PRED_FB, shape.output_frames, "list")]
    assert len(adapter.calls) == shape.output_frames
    assert all(call["payload_bits"] == 64 for call in adapter.calls)
    assert result.total_latency_ms == pytest.approx(6.6)


def test_parallel_runs_one_frame_ue_then_air_then_user_bs_part():
    shape = make_shape()
    context = make_context(shape)
    ue = CountingStep(out_features=8)
    bs = CountingBS()
    network = ParallelNetwork(name="parallel-unit", encoder=ue, bs_network=bs, feedback_frames=1)
    flow = network.build_flow(shape)
    adapter = FakeAdapter()
    result = FlowExecutor(FakeProfiler([0.2, 0.7]), adapter, FeedbackSpec(bitwidth=8)).run(flow, context)

    assert ue.calls == 1
    assert bs.calls == [(FlowKind.PARALLEL, 1, "Tensor")]
    assert len(adapter.calls) == 1
    assert result.total_latency_ms == pytest.approx(2.15)
    assert {timing.kind for timing in result.stage_timings} == {
        StageKind.UE_MODEL,
        StageKind.AIR_FEEDBACK,
        StageKind.BS_MODEL,
    }


def test_same_one_frame_feedback_uses_same_scheduling_request_time():
    shape = make_shape()
    context = make_context(shape)
    bs = CountingBS()

    fb_flow = SerialNetwork.fb_pred(
        name="fb-pred-unit",
        encoder=CountingStep(out_features=8),
        bs_steps=[("bs_network", bs)],
        feedback_frames=1,
    ).build_flow(shape)
    parallel_flow = ParallelNetwork(
        name="parallel-unit",
        encoder=CountingStep(out_features=8),
        bs_network=bs,
        feedback_frames=1,
    ).build_flow(shape)

    fb_adapter = FakeAdapter()
    parallel_adapter = FakeAdapter()
    FlowExecutor(FakeProfiler([0.2, 1.0]), fb_adapter, FeedbackSpec(bitwidth=8)).run(fb_flow, context)
    FlowExecutor(FakeProfiler([2.0, 1.0]), parallel_adapter, FeedbackSpec(bitwidth=8)).run(parallel_flow, context)

    assert fb_adapter.calls == [{"ue_id": 1, "request_time_ms": 0.0, "payload_bits": 64}]
    assert parallel_adapter.calls == [{"ue_id": 1, "request_time_ms": 0.0, "payload_bits": 64}]


def test_explicit_payload_bits_override_tensor_size():
    shape = make_shape()
    context = make_context(shape)
    bs = CountingBS()

    class PayloadUE(nn.Module):
        name = "payload_ue"

        def forward(self, window: CSIWindow):
            return {"payload_bits": 130, "latent": torch.ones(1024)}

    flow = ParallelNetwork("payload-unit", PayloadUE(), bs).build_flow(shape)
    adapter = FakeAdapter()
    FlowExecutor(FakeProfiler([0.2, 0.7]), adapter, FeedbackSpec(bitwidth=8)).run(flow, context)

    assert adapter.calls[0]["payload_bits"] == 130


def test_network_api_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        ParallelNetwork(name="", encoder=lambda x: x, bs_network=lambda x: x)
    with pytest.raises(TypeError):
        ParallelNetwork(name="bad", encoder=None, bs_network=lambda x: x)
    with pytest.raises(ValueError):
        ParallelNetwork(name="bad", encoder=lambda x: x, bs_network=lambda x: x, feedback_frames=0)
    with pytest.raises(ValueError):
        SerialNetwork.pred_fb(name="bad", bs_steps=[("bs", lambda x: x)])
    with pytest.raises(ValueError):
        ModelStep("bad", lambda x: x, repeat=0)


def test_lstm_mlp_benchmark_matches_flow_contract():
    shape = make_shape()
    context = make_context(shape)
    for flow in [FlowKind.FB_PRED, FlowKind.PRED_FB, FlowKind.PARALLEL]:
        network = build_lstm_mlp_network(shape, flow, latent_dim=8, hidden_dim=16)
        flow_model = network.build_flow(shape)
        result, _, _ = FlowExecutor(
            FakeProfiler([0.01] * 16),
            FakeAdapter(),
            FeedbackSpec(bitwidth=8),
        )._measure_steps(
            flow_model.bs_steps,
            _bs_input_for_contract(flow_model, context),
            context,
            StageKind.BS_MODEL,
            side=ExecutionSide.BS,
            start_ms=0.0,
            feedback_frames=flow_model.resolved_feedback_frames(shape),
        )
        assert tuple(result.shape) == (shape.output_frames, *shape.frame_shape)


def _bs_input_for_contract(flow_model, context):
    if flow_model.flow == FlowKind.PRED_FB:
        return BSInput(
            context.window,
            [torch.ones(8) for _ in range(context.shape.output_frames)],
            flow_model.flow,
            context.shape.output_frames,
        )
    return BSInput(context.window, torch.ones(8), flow_model.flow, 1)


def make_shape() -> CSIShapeSpec:
    return CSIShapeSpec(
        frame_shape=(2, 4, 4),
        axes=("complex", "rx_ant", "subcarrier"),
        history_len=4,
        horizon=3,
        dtype="float32",
    )


def make_context(shape: CSIShapeSpec) -> CPLEContext:
    history = torch.randn(shape.history_len, *shape.frame_shape)
    window = CSIWindow(
        history=history,
        current=history[-1],
        target=torch.randn(shape.output_frames, *shape.frame_shape),
    )
    return CPLEContext(
        run_id="unit",
        slot_idx=0,
        tti_ms=1.0,
        ue_id=1,
        bs_id=0,
        device="cpu",
        shape=shape,
        window=window,
    )
