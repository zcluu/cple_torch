# CPLE

CPLE (**Communication Process Lightweight Evaluation**) is a runtime evaluation framework for PyTorch-based CSI feedback and prediction pipelines under a Sionna SYS communication simulation flow.

The current version focuses on runtime behavior that is hard to capture with static metrics such as FLOPs or parameter count. It compares serial and parallel CSI pipelines under a slot-based communication workload and reports model computation latency, feedback scheduling delay, feedback occupation time, and per-stage runtime.

## What CPLE Evaluates

CPLE is designed around this comparison:

```text
Serial pipeline:
  predict future CSI frames
  -> feedback frame 0
  -> feedback frame 1
  -> feedback frame 2
  -> feedback frame 3

Parallel pipeline:
  feedback once
  -> BS-side joint prediction
  -> output T=0..P in one forward pass
```

The framework uses Sionna SYS to simulate UE scheduling and OFDM resource allocation. CPLE consumes the simulated resources and computes latency metrics itself:

```text
total latency = model computation latency
              + feedback scheduling delay
              + feedback resource occupation time
```

This makes the serial pipeline pay for repeated feedback resource usage, while the parallel pipeline only occupies feedback resources once per service request.

## Repository Contents

- `cple/`: core CPLE package
- `configs/`: runnable experiment and Sionna-style scenario configs
- `examples/`: smoke run and user-model integration example
- `tests/`: regression tests
- `setup.py`: package installation entry
- `requirements.txt`: runtime dependencies

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

For tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Quick Start

Run the built-in smoke experiment:

```powershell
.\.venv\Scripts\python.exe examples\run_smoke.py
```

Run the user-model example:

```powershell
.\.venv\Scripts\python.exe examples\run_user_models.py --config configs\user_models.yaml
```

Validate scenario configs:

```powershell
.\.venv\Scripts\python.exe -m cple.tools.validate_scenarios configs\sionna_umi_low_mobility.yaml
```

## Outputs

Current runs export:

- `latency_summary.csv`
- `stage_summary.csv`
- `environment.txt`

`latency_summary.csv` includes model latency, feedback scheduling delay, feedback duration, total latency, deadline miss ratio, and per-output total latency.

`stage_summary.csv` reports per-stage runtime for serial pipelines.

## Current Scope

Implemented:

- API-first user model interface
- Serial and parallel model execution
- Per-stage runtime profiling
- Sionna SYS proportional-fair scheduling through `PFSchedulerSUMIMO`
- Sionna topology sampling through `gen_hexgrid_topology`
- 3GPP TR 38.901 channel coefficient generation for UMi/UMa/RMa profiles
- Scenario YAML loading and validation
- Example PyTorch CSI models
- Regression tests

Not implemented yet:

- SINR/BLER/MCS and PHY abstraction integration
- Real checkpoint loading for project-specific CSI models
- Feedback payload bit-level modeling
- Asynchronous serving
