from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass

from ..api import BSInput, CSIShapeSpec, CSIWindow, FlowKind, ParallelNetwork, SerialNetwork


@dataclass
class DecodedFrame:
    frame: torch.Tensor
    source: BSInput


@dataclass
class PredictedFuture:
    current: torch.Tensor
    future: torch.Tensor

    def as_result(self) -> torch.Tensor:
        return torch.cat([self.current.reshape(1, *self.current.shape), self.future], dim=0)


class MLPCodec(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(torch.float32))


class CSIEncoder(nn.Module):
    name = "encoder"

    def __init__(self, shape: CSIShapeSpec, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.shape = shape
        self.codec = MLPCodec(shape.elements_per_frame, latent_dim, hidden_dim)

    def forward(self, data) -> torch.Tensor:
        if isinstance(data, CSIWindow):
            x = data.current.reshape(-1)
        elif isinstance(data, torch.Tensor):
            x = data.reshape(-1)[: self.shape.elements_per_frame]
        else:
            x = data.window.current.reshape(-1)
        return self.codec(x)


class CSIDecoder(nn.Module):
    name = "decoder"

    def __init__(self, shape: CSIShapeSpec, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.shape = shape
        self.codec = MLPCodec(latent_dim, shape.elements_per_frame, hidden_dim)

    def forward(self, data) -> torch.Tensor:
        if isinstance(data, BSInput):
            if isinstance(data.ue_output, (list, tuple)):
                frames = [self.codec(item.reshape(-1)).reshape(self.shape.frame_shape) for item in data.ue_output]
                return torch.stack(frames, dim=0)
            x = data.ue_output
        else:
            x = data
        frame = self.codec(x.reshape(-1)).reshape(self.shape.frame_shape)
        if isinstance(data, BSInput):
            return DecodedFrame(frame=frame, source=data)
        return frame


class LSTMPredictor(nn.Module):
    name = "predictor"

    def __init__(self, shape: CSIShapeSpec, hidden_dim: int):
        super().__init__()
        self.shape = shape
        self.lstm = nn.LSTM(
            input_size=shape.elements_per_frame,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, shape.horizon * shape.elements_per_frame)
        self.eval()

    def forward(self, data) -> torch.Tensor | PredictedFuture:
        current = None
        if isinstance(data, CSIWindow):
            history = data.history
        elif isinstance(data, BSInput):
            history = data.window.history
        elif isinstance(data, DecodedFrame):
            current = data.frame
            decoded_current = current.reshape(1, *self.shape.frame_shape)
            history = torch.cat([data.source.window.history[1:], decoded_current], dim=0)
        else:
            flat = data.reshape(-1)
            if flat.numel() == self.shape.elements_per_frame:
                history = flat.reshape(1, *self.shape.frame_shape).repeat(self.shape.history_len, *([1] * len(self.shape.frame_shape)))
            else:
                history = flat.reshape(self.shape.history_len, *self.shape.frame_shape)
        x = history.reshape(1, self.shape.history_len, self.shape.elements_per_frame).to(torch.float32)
        _, (hidden, _) = self.lstm(x)
        future = self.head(hidden[-1]).reshape(self.shape.horizon, *self.shape.frame_shape)
        if current is not None:
            return PredictedFuture(current=current, future=future)
        return future


class FBPredAssembler(nn.Module):
    name = "assemble_result"

    def forward(self, data: PredictedFuture | torch.Tensor) -> torch.Tensor:
        if isinstance(data, PredictedFuture):
            return data.as_result()
        return data


class ParallelBSNetwork(nn.Module):
    name = "bs_network"

    def __init__(self, shape: CSIShapeSpec, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.shape = shape
        self.decoder = CSIDecoder(shape, latent_dim, hidden_dim)
        self.predictor = LSTMPredictor(shape, hidden_dim)

    def forward(self, data: BSInput) -> torch.Tensor:
        decoded = self.decoder(data)
        decoded_current = decoded.frame.reshape(1, *self.shape.frame_shape)
        history = torch.cat([data.window.history[1:], decoded_current], dim=0)
        predicted = self.predictor(
            CSIWindow(
                history=history,
                current=decoded_current[0],
                target=data.window.target,
                raw=data.window.raw,
            )
        )
        return torch.cat([decoded_current, predicted], dim=0)


def build_lstm_mlp_network(
    shape: CSIShapeSpec,
    flow: str | FlowKind,
    *,
    latent_dim: int = 64,
    hidden_dim: int = 128,
    device: str = "cpu",
) -> SerialNetwork | ParallelNetwork:
    flow_kind = FlowKind(flow)
    name = f"lstm_mlp_{flow_kind.value.replace('-', '_')}"
    if flow_kind == FlowKind.FB_PRED:
        return SerialNetwork.fb_pred(
            name=name,
            encoder=CSIEncoder(shape, latent_dim, hidden_dim).to(device),
            bs_steps=[
                ("decoder", CSIDecoder(shape, latent_dim, hidden_dim).to(device)),
                ("predictor", LSTMPredictor(shape, hidden_dim).to(device)),
                ("assemble_result", FBPredAssembler().to(device)),
            ],
            feedback_frames=1,
            prediction_frames=shape.horizon,
        )
    if flow_kind == FlowKind.PRED_FB:
        return SerialNetwork.pred_fb(
            name=name,
            predictor=LSTMPredictor(shape, hidden_dim).to(device),
            encoder=CSIEncoder(shape, latent_dim, hidden_dim).to(device),
            bs_steps=[("decoder", CSIDecoder(shape, latent_dim, hidden_dim).to(device))],
            feedback_frames=shape.output_frames,
            prediction_frames=shape.horizon,
        )
    return ParallelNetwork(
        name=name,
        encoder=CSIEncoder(shape, latent_dim, hidden_dim).to(device),
        bs_network=ParallelBSNetwork(shape, latent_dim, hidden_dim).to(device),
        feedback_frames=1,
        prediction_frames=shape.horizon,
    )
