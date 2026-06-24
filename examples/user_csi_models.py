from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cple import BSInput, CSIShapeSpec, CSIWindow, FlowKind, ParallelNetwork, SerialNetwork


class UserUEPart(nn.Module):
    """UE-side model part, e.g. CSI/codeword encoder."""

    name = "user_ue_part"

    def __init__(self, shape: CSIShapeSpec, feedback_frames: int, compress_ratio: float = 2.0):
        super().__init__()
        self.shape = shape
        self.feedback_frames = feedback_frames
        in_features = shape.elements_per_frame * feedback_frames
        out_features = max(1, int(in_features / compress_ratio))
        self.net = nn.Linear(in_features, out_features)
        self.eval()

    def forward(self, data) -> torch.Tensor:
        if isinstance(data, CSIWindow):
            x = data.current.reshape(-1)
        elif isinstance(data, torch.Tensor):
            x = data.reshape(-1)
        else:
            x = data.window.current.reshape(-1)
        expected = self.shape.elements_per_frame * self.feedback_frames
        if x.numel() < expected:
            x = torch.nn.functional.pad(x, (0, expected - x.numel()))
        else:
            x = x[:expected]
        return self.net(x.to(torch.float32))


class UserPredictor(nn.Module):
    """UE-side predictor returning future CSI frames."""

    name = "user_predictor"

    def __init__(self, shape: CSIShapeSpec):
        super().__init__()
        self.shape = shape
        self.net = nn.Linear(
            shape.history_len * shape.elements_per_frame,
            shape.horizon * shape.elements_per_frame,
        )
        self.eval()

    def forward(self, window: CSIWindow) -> torch.Tensor:
        x = window.history.reshape(-1).to(torch.float32)
        return self.net(x).reshape(self.shape.horizon, *self.shape.frame_shape)


class UserBSPart(nn.Module):
    """BS-side model part; prediction/decoding order is internal to this module."""

    name = "user_bs_part"

    def __init__(self, shape: CSIShapeSpec):
        super().__init__()
        self.shape = shape
        self.net = nn.Sequential(
            nn.Linear(max(1, shape.elements_per_frame * shape.output_frames), shape.elements_per_frame),
            nn.ReLU(),
            nn.Linear(shape.elements_per_frame, shape.output_frames * shape.elements_per_frame),
        )
        self.eval()

    def forward(self, data: BSInput) -> torch.Tensor:
        if isinstance(data.ue_output, (list, tuple)):
            x = torch.cat([item.reshape(-1).to(torch.float32) for item in data.ue_output])
        elif isinstance(data.ue_output, torch.Tensor):
            x = data.ue_output.reshape(-1).to(torch.float32)
        else:
            x = data.window.current.reshape(-1).to(torch.float32)
        if x.numel() < self.net[0].in_features:
            x = torch.nn.functional.pad(x, (0, self.net[0].in_features - x.numel()))
        else:
            x = x[: self.net[0].in_features]
        return self.net(x).reshape(data.window.target.shape)


def build_user_flow(
    shape: CSIShapeSpec,
    device: str = "cpu",
    flow: str | FlowKind = FlowKind.FB_PRED,
) -> SerialNetwork | ParallelNetwork:
    flow_kind = FlowKind(flow)
    name = f"user_{flow_kind.value.replace('-', '_')}"
    bs_part = UserBSPart(shape).to(device)
    if flow_kind == FlowKind.FB_PRED:
        return SerialNetwork.fb_pred(
            name=name,
            encoder=UserUEPart(shape, feedback_frames=1).to(device),
            bs_steps=[("bs_network", bs_part)],
            feedback_frames=1,
            prediction_frames=shape.horizon,
        )
    if flow_kind == FlowKind.PRED_FB:
        return SerialNetwork.pred_fb(
            name=name,
            predictor=UserPredictor(shape).to(device),
            encoder=UserUEPart(shape, feedback_frames=1).to(device),
            bs_steps=[("bs_network", bs_part)],
            feedback_frames=shape.output_frames,
            prediction_frames=shape.horizon,
        )
    return ParallelNetwork(
        name=name,
        encoder=UserUEPart(shape, feedback_frames=1).to(device),
        bs_network=bs_part,
        feedback_frames=1,
        prediction_frames=shape.horizon,
    )


if __name__ == "__main__":
    spec = CSIShapeSpec(
        frame_shape=(2, 4, 4),
        axes=("complex", "rx_ant", "subcarrier"),
        history_len=4,
        horizon=3,
        dtype="float32",
    )
    network = build_user_flow(spec, flow=FlowKind.PARALLEL)
    flow = network.build_flow(spec)
    window = CSIWindow(
        history=torch.randn(spec.history_len, *spec.frame_shape),
        current=torch.randn(*spec.frame_shape),
        target=torch.randn(spec.output_frames, *spec.frame_shape),
    )
    ue_output = flow.ue_steps[0].module(window)
    assert ue_output.ndim == 1
    assert flow.bs_steps[0].module(BSInput(window, ue_output, flow.flow, 1)).shape == window.target.shape
    print("user CPLE UE/BS parts are importable")
