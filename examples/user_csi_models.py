from __future__ import annotations

import torch
from torch import nn

from cple import (
    CPLEParallelModelAPI,
    CPLESerialModelAPI,
    CPLEStage,
    CSIServiceResult,
    ModelCapability,
)


class UserSerialPredictThenFeedback(CPLESerialModelAPI):
    """User-defined serial CSI pipeline.

    Flow:
      1. UE-side model predicts P future CSI frames.
      2. The predicted frames are fed back one by one to the BS side.

    This class is intentionally simple: two Linear layers stand in for the
    user's real PyTorch CSI prediction and feedback modules.
    """

    name = "user_serial_predict_then_feedback"

    def __init__(self, csi_dim: int = 16, horizon: int = 3):
        self.csi_dim = csi_dim
        self.horizon = horizon
        self.predictor = nn.Linear(csi_dim, csi_dim * horizon)
        self.feedback = nn.Sequential(
            nn.Linear(csi_dim, csi_dim // 2),
            nn.ReLU(),
            nn.Linear(csi_dim // 2, csi_dim),
        )
        self.predictor.eval()
        self.feedback.eval()

    def prepare_input(self, context):
        return context.h_t.float()

    def forward(self, model_input):
        predicted_bundle = self.predict_future_p_frames(model_input)
        outputs = {"predict_future_p_frames": predicted_bundle}
        for frame_idx in range(self.horizon + 1):
            outputs[f"feedback_frame_{frame_idx}"] = self.feedback_frame(
                predicted_bundle, frame_idx
            )
        return outputs

    def predict_future_p_frames(self, model_input):
        future = self.predictor(model_input).reshape(self.horizon, self.csi_dim)
        return torch.cat([model_input.reshape(1, self.csi_dim), future], dim=0)

    def feedback_frame(self, frame_bundle, frame_idx: int):
        return self.feedback(frame_bundle[frame_idx])

    def stages(self):
        stages = [
            CPLEStage(
                "predict_future_p_frames",
                self.predict_future_p_frames,
                output_frames=list(range(1, self.horizon + 1)),
                operation_type="prediction",
            ),
        ]
        previous = "predict_future_p_frames"
        for frame_idx in range(self.horizon + 1):
            stage_name = f"feedback_frame_{frame_idx}"
            stages.append(
                CPLEStage(
                    stage_name,
                    lambda bundle, idx=frame_idx: self.feedback_frame(bundle, idx),
                    depends_on=[previous],
                    input_from="predict_future_p_frames",
                    output_frames=[frame_idx],
                    operation_type="feedback",
                )
            )
            previous = stage_name
        return stages

    def parse_output(self, output, context):
        feedback_payload = torch.stack(
            [output[f"feedback_frame_{idx}"] for idx in range(self.horizon + 1)],
            dim=0,
        )
        frames = {
            idx: feedback_payload[idx].detach().clone()
            for idx in range(self.horizon + 1)
        }
        return CSIServiceResult(
            frames=frames,
            valid_from_slot=context.slot_idx,
            valid_until_slot=context.slot_idx + self.horizon,
            metadata={"flow": "serial_predict_then_feedback"},
        )

    def capability(self):
        return ModelCapability(
            output_frames=list(range(self.horizon + 1)),
            requires_history=0,
            feedback_requests=self.horizon + 1,
        )


class UserParallelFeedbackThenPredict(CPLEParallelModelAPI):
    """User-defined parallel/network-side CSI pipeline.

    Flow:
      1. UE feeds back the current compressed/reconstructed CSI representation.
      2. BS-side model performs one forward pass and emits T=0 plus T=1..P.

    One Linear layer stands in for the real network-side joint feedback and
    prediction model.
    """

    name = "user_parallel_feedback_then_predict"

    def __init__(self, csi_dim: int = 16, horizon: int = 3):
        self.csi_dim = csi_dim
        self.horizon = horizon
        self.joint_feedback_predictor = nn.Linear(csi_dim, csi_dim * (horizon + 1))
        self.joint_feedback_predictor.eval()

    def prepare_input(self, context):
        return context.h_t.float()

    def forward(self, model_input):
        return self.joint_feedback_predictor(model_input)

    def parse_output(self, output, context):
        frames_tensor = output.reshape(self.horizon + 1, self.csi_dim)
        frames = {
            idx: frames_tensor[idx].detach().clone() for idx in range(self.horizon + 1)
        }
        return CSIServiceResult(
            frames=frames,
            valid_from_slot=context.slot_idx,
            valid_until_slot=context.slot_idx + self.horizon,
            metadata={"flow": "parallel_feedback_then_predict"},
        )

    def capability(self):
        return ModelCapability(
            output_frames=list(range(self.horizon + 1)),
            requires_history=0,
            feedback_requests=1,
        )


def build_user_models(csi_dim: int = 16, horizon: int = 3):
    return [
        UserSerialPredictThenFeedback(csi_dim=csi_dim, horizon=horizon),
        UserParallelFeedbackThenPredict(csi_dim=csi_dim, horizon=horizon),
    ]


if __name__ == "__main__":
    # Tiny sanity check for developers editing this example directly.
    x = torch.randn(16)
    serial = UserSerialPredictThenFeedback(csi_dim=16)
    parallel = UserParallelFeedbackThenPredict(csi_dim=16)
    bundle = serial.predict_future_p_frames(x)
    assert bundle.shape == (4, 16)
    assert torch.stack(
        [serial.feedback_frame(bundle, idx) for idx in range(4)]
    ).shape == (4, 16)
    assert parallel.forward(x).shape == (64,)
    print("user CSI models are importable")
