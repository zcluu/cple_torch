from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cple.configs.runner import build_adapter, load_config
from cple.models import build_lstm_mlp_network
from cple.runtime.platform import CPLEPlatform


def run_one(config, flow: str, latent_dim: int, hidden_dim: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    torch.manual_seed(config.platform.seed)
    adapter = build_adapter(config)
    shape = config.shape.to_spec()
    network = build_lstm_mlp_network(
        shape,
        flow,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        device=config.platform.device,
    )
    platform = CPLEPlatform(config.platform, shape, config.feedback, adapter)
    summaries = platform.run(network)
    return summaries["latency_summary"], summaries["stage_summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fb-pred, pred-fb, and parallel using LSTM predictor + MLP codec.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/lstm_mlp_compare")
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    config = load_config(args.config)
    latency_frames = []
    stage_frames = []
    for flow in ["fb-pred", "pred-fb", "parallel"]:
        latency, stage = run_one(config, flow, args.latent_dim, args.hidden_dim)
        latency_frames.append(latency)
        stage_frames.append(stage)

    latency_all = pd.concat(latency_frames, ignore_index=True)
    stage_all = pd.concat(stage_frames, ignore_index=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latency_all.to_csv(output_dir / "latency_summary.csv", index=False)
    stage_all.to_csv(output_dir / "stage_summary.csv", index=False)
    cols = [
        "flow_kind",
        "mean_model_runtime_ms",
        "mean_scheduling_delay_ms",
        "mean_feedback_duration_ms",
        "mean_total_latency_ms",
        "deadline_miss_ratio",
    ]
    print("\nLatency summary")
    print(latency_all[cols].to_string(index=False))

    print("\nStage summary")
    stage_cols = ["flow_kind", "stage_name", "side", "event_count", "mean_stage_ms", "mean_payload_bits"]
    print(stage_all[stage_cols].to_string(index=False))
    print(f"\nSaved: {output_dir / 'latency_summary.csv'}")
    print(f"Saved: {output_dir / 'stage_summary.csv'}")


if __name__ == "__main__":
    main()
