# CPLE Torch

CPLE Torch is a PyTorch-based latency evaluation toolkit for CSI feedback and
prediction pipelines. It measures the runtime of UE-side models, feedback
scheduling and occupation time, and BS-side models under a slot-based wireless
service process.

CPLE Torch is focused on latency and scheduling behavior. It does not evaluate
prediction accuracy.

## Features

- User-defined CSI tensor shape, history length, and prediction horizon.
- Three feedback/prediction strategies: `fb-pred`, `pred-fb`, and `parallel`.
- PyTorch module integration through simple callable APIs.
- Sionna SYS-backed scheduling and channel-window generation.
- Per-stage runtime profiling for CPU and CUDA.
- CSV outputs for raw events, latency summaries, and stage summaries.
- MkDocs documentation with generated API reference.

## Strategy Overview

For prediction horizon `P`, each strategy targets `P + 1` final frames:

| Strategy | UE-side work | Feedback channel | BS-side work |
| --- | --- | --- | --- |
| `fb-pred` | Encode current CSI/codeword | One feedback frame | Decode one frame, then predict `P` frames |
| `pred-fb` | Predict `P` frames, then encode `P + 1` frames | `P + 1` feedback frames | Decode `P + 1` frames |
| `parallel` | Encode current CSI/codeword | One feedback frame | Run a user-provided BS-side network |

See [docs/concepts/flow-io.md](docs/concepts/flow-io.md) for the full module
input/output contract.

## Installation

Install from the repository root:

```bash
python -m pip install .
```

For editable development:

```bash
python -m pip install -e ".[dev]"
```

For documentation builds:

```bash
python -m pip install -e ".[docs]"
```

## Quickstart

Run the smoke example:

```bash
python examples/run_smoke.py
```

Compare the built-in LSTM + MLP benchmark models:

```bash
python examples/compare_lstm_mlp.py --config configs/smoke.yaml --torch-threads 1
```

Validate a Sionna scenario profile:

```bash
python -m cple.tools.validate_scenarios configs/sionna_umi_low_mobility.yaml
```

## Basic API

```python
from cple import ParallelNetwork

network = ParallelNetwork(
    name="my_parallel_network",
    encoder=ue_encoder,
    bs_network=bs_network,
    feedback_frames=1,
    prediction_frames=3,
)

summaries = platform.run(network)
```

Serial strategies use `SerialNetwork`:

```python
from cple import SerialNetwork

network = SerialNetwork.pred_fb(
    name="my_pred_fb",
    predictor=ue_predictor,
    encoder=ue_encoder,
    bs_steps=[("decoder", bs_decoder)],
    prediction_frames=3,
)
```

## Configuration

Run-level parameters are defined in YAML:

```yaml
sionna_scenario_path: sionna_umi_low_mobility.yaml
platform:
  run_id: smoke
  num_slots: 20
  deadline_ms: 5.0
  device: cpu
  warmup_slots: 1
shape:
  frame_shape: [2, 4, 4]
  axes: [complex, rx_ant, subcarrier]
  history_len: 4
  horizon: 3
feedback:
  bitwidth: 8
  bits_per_resource_unit: 64
adapter:
  scheduled_per_slot: 2
flow: fb-pred
```

## Outputs

CPLE Torch writes:

- `event_log.csv`: raw slot, flow, stage, payload, and timing events.
- `latency_summary.csv`: per-flow latency statistics.
- `stage_summary.csv`: per-stage runtime and feedback-resource statistics.
- `environment.txt`: runtime environment metadata.

## Documentation

Build the documentation:

```bash
mkdocs build
```

Serve locally:

```bash
mkdocs serve
```

The documentation source is in [docs/](docs/) and the MkDocs configuration is
in [mkdocs.yml](mkdocs.yml).

## Testing

```bash
python -m pytest -q
```

## License

MIT
