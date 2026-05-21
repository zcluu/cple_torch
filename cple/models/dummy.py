from __future__ import annotations

import torch
from torch import nn

from ..api import (
    CPLEContext,
    CPLEParallelModelAPI,
    CPLESerialModelAPI,
    CPLEStage,
    CSIServiceResult,
    ModelCapability,
)


class DummyParallelModel(CPLEParallelModelAPI):
    name = "dummy_parallel"

    def __init__(self, csi_dim: int = 8, horizon: int = 3):
        self.csi_dim = csi_dim
        self.horizon = horizon
        self.net = nn.Linear(csi_dim, csi_dim * (horizon + 1))
        self.net.eval()

    def prepare_input(self, context: CPLEContext):
        return context.h_t.float()

    def forward(self, model_input):
        return self.net(model_input)

    def parse_output(self, output, context: CPLEContext) -> CSIServiceResult:
        chunks = output.reshape(self.horizon + 1, self.csi_dim)
        frames = {idx: chunks[idx].detach().clone() for idx in range(self.horizon + 1)}
        return CSIServiceResult(
            frames=frames,
            valid_from_slot=context.slot_idx,
            valid_until_slot=context.slot_idx + self.horizon,
            metadata={"model_type": "dummy_parallel"},
        )

    def capability(self) -> ModelCapability:
        return ModelCapability(
            output_frames=list(range(self.horizon + 1)),
            requires_history=0,
            feedback_requests=1,
        )


class DummySerialModel(CPLESerialModelAPI):
    name = "dummy_serial"

    def __init__(self, csi_dim: int = 8, horizon: int = 3):
        self.csi_dim = csi_dim
        self.horizon = horizon
        self.feedback = nn.Linear(csi_dim, csi_dim)
        self.prediction = nn.Linear(csi_dim, csi_dim * horizon)
        self.feedback.eval()
        self.prediction.eval()

    def prepare_input(self, context: CPLEContext):
        return context.h_t.float()

    def forward(self, model_input):
        reconstructed = self.feedback(model_input)
        future = self.prediction(reconstructed)
        return {"feedback": reconstructed, "prediction": future}

    def run_feedback(self, model_input):
        return self.feedback(model_input)

    def run_prediction(self, reconstructed):
        return self.prediction(reconstructed)

    def stages(self) -> list[CPLEStage]:
        return [
            CPLEStage("feedback_reconstruction", self.run_feedback, output_frames=[0], operation_type="feedback"),
            CPLEStage(
                "future_prediction",
                self.run_prediction,
                depends_on=["feedback_reconstruction"],
                output_frames=list(range(1, self.horizon + 1)),
                operation_type="prediction",
            ),
        ]

    def parse_output(self, output, context: CPLEContext) -> CSIServiceResult:
        reconstructed = output["feedback_reconstruction"].detach().clone()
        future = output["future_prediction"].reshape(self.horizon, self.csi_dim)
        frames = {0: reconstructed}
        for idx in range(1, self.horizon + 1):
            frames[idx] = future[idx - 1].detach().clone()
        return CSIServiceResult(
            frames=frames,
            valid_from_slot=context.slot_idx,
            valid_until_slot=context.slot_idx + self.horizon,
            metadata={"model_type": "dummy_serial"},
        )

    def capability(self) -> ModelCapability:
        return ModelCapability(
            output_frames=list(range(self.horizon + 1)),
            requires_history=0,
            feedback_requests=1,
        )
