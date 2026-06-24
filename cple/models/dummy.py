from __future__ import annotations

import torch
from torch import nn

from ..api import BSInput, CSIShapeSpec, CSIWindow, FlowKind, ParallelNetwork, SerialNetwork


class LinearUEPart(nn.Module):
    name = "linear_ue_part"

    def __init__(self, shape: CSIShapeSpec, frames: int, latent_ratio: float = 2.0):
        super().__init__()
        self.shape = shape
        self.frames = frames
        self.out_features = max(1, int(shape.elements_per_frame * frames / latent_ratio))
        self.net = nn.Linear(shape.elements_per_frame * frames, self.out_features)
        self.eval()

    def forward(self, data) -> torch.Tensor:
        if isinstance(data, CSIWindow):
            x = data.current.reshape(-1) if self.frames == 1 else data.history[-self.frames :].reshape(-1)
        elif isinstance(data, torch.Tensor):
            x = data.reshape(-1)
        else:
            x = data.window.current.reshape(-1)
        expected = self.shape.elements_per_frame * self.frames
        if x.numel() < expected:
            x = torch.nn.functional.pad(x, (0, expected - x.numel()))
        else:
            x = x[:expected]
        return self.net(x.to(torch.float32))


class LinearPredictor(nn.Module):
    name = "linear_predictor"

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


class LinearBSPart(nn.Module):
    name = "linear_bs_part"

    def __init__(self, shape: CSIShapeSpec):
        super().__init__()
        self.shape = shape
        self.in_features = shape.output_frames * shape.elements_per_frame
        self.net = nn.Linear(
            self.in_features,
            shape.output_frames * shape.elements_per_frame,
        )
        self.eval()

    def forward(self, data: BSInput) -> torch.Tensor:
        if isinstance(data.ue_output, (list, tuple)):
            x = torch.cat([item.reshape(-1).to(torch.float32) for item in data.ue_output])
        elif isinstance(data.ue_output, torch.Tensor):
            x = data.ue_output.reshape(-1).to(torch.float32)
        else:
            x = data.window.current.reshape(-1).to(torch.float32)
        if x.numel() < self.in_features:
            x = torch.nn.functional.pad(x, (0, self.in_features - x.numel()))
        else:
            x = x[: self.in_features]
        return self.net(x).reshape(data.window.target.shape)


def build_dummy_flow(
    shape: CSIShapeSpec,
    device: str = "cpu",
    flow: str | FlowKind = FlowKind.FB_PRED,
) -> SerialNetwork | ParallelNetwork:
    flow_kind = FlowKind(flow)
    name = f"dummy_{flow_kind.value.replace('-', '_')}"
    bs_part = LinearBSPart(shape).to(device)
    if flow_kind == FlowKind.FB_PRED:
        return SerialNetwork.fb_pred(
            name=name,
            encoder=LinearUEPart(shape, frames=1).to(device),
            bs_steps=[("bs_network", bs_part)],
            feedback_frames=1,
            prediction_frames=shape.horizon,
        )
    if flow_kind == FlowKind.PRED_FB:
        return SerialNetwork.pred_fb(
            name=name,
            predictor=LinearPredictor(shape).to(device),
            encoder=LinearUEPart(shape, frames=1).to(device),
            bs_steps=[("bs_network", bs_part)],
            feedback_frames=shape.output_frames,
            prediction_frames=shape.horizon,
        )
    return ParallelNetwork(
        name=name,
        encoder=LinearUEPart(shape, frames=1).to(device),
        bs_network=bs_part,
        feedback_frames=1,
        prediction_frames=shape.horizon,
    )
