# Configuration

CPLE uses YAML configuration for run-level parameters.

## Experiment Config

```yaml
sionna_scenario_path: sionna_umi_low_mobility.yaml
platform:
  run_id: smoke
  num_slots: 20
  deadline_ms: 5.0
  device: cpu
  warmup_slots: 1
  output_dir: outputs/smoke
  seed: 7
shape:
  frame_shape: [2, 4, 4]
  axes: [complex, rx_ant, subcarrier]
  history_len: 4
  horizon: 3
  dtype: float32
feedback:
  bitwidth: 8
  bits_per_resource_unit: 64
adapter:
  scheduled_per_slot: 2
  seed: 7
flow: fb-pred
```

## Platform Fields

| Field | Meaning |
| --- | --- |
| `run_id` | Identifier written to event logs |
| `num_slots` | Number of simulated service slots |
| `tti_ms` | Slot duration in milliseconds |
| `deadline_ms` | Deadline used for miss ratio metrics |
| `device` | Profiling/model device |
| `warmup_slots` | Initial slots excluded from metrics |
| `output_dir` | Output directory for CSV files |
| `seed` | Runtime seed |

## Shape Fields

| Field | Meaning |
| --- | --- |
| `frame_shape` | Per-frame CSI tensor shape |
| `axes` | Optional axis names |
| `history_len` | Number of history frames |
| `horizon` | Number of future frames to predict |
| `dtype` | `float32` or `complex64` |

## Feedback Fields

| Field | Meaning |
| --- | --- |
| `bitwidth` | Bits per tensor element for payload estimation |
| `bits_per_resource_unit` | Payload capacity per feedback resource unit |

`bitwidth` is used for payload-size estimation. It does not simulate
quantization.

## Adapter Fields

| Field | Meaning |
| --- | --- |
| `num_ues` | Number of UEs in the adapter |
| `scheduled_per_slot` | Maximum UEs selected by CPLE per slot |
| `seed` | Adapter seed |

When `sionna_scenario_path` is provided, scenario values are loaded first and
the experiment config overrides selected fields.
