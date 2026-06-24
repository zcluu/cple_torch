# Quickstart

## Run The Smoke Example

```bash
python examples/run_smoke.py
```

This uses `configs/smoke.yaml`, runs the dummy model flow, and writes:

```text
outputs/smoke/event_log.csv
outputs/smoke/latency_summary.csv
outputs/smoke/stage_summary.csv
outputs/smoke/environment.txt
```

## Run User Model Example

```bash
python examples/run_user_models.py --config configs/user_models.yaml
```

This demonstrates passing user-provided UE and BS modules into CPLE.

## Compare LSTM + MLP Benchmark Models

```bash
python examples/compare_lstm_mlp.py --config configs/smoke.yaml --torch-threads 1
```

The script runs `fb-pred`, `pred-fb`, and `parallel` independently and combines
their summaries under `outputs/lstm_mlp_compare/`.

## Validate Scenario Profiles

```bash
python -m cple.tools.validate_scenarios configs/sionna_umi_low_mobility.yaml
```

The command checks scenario schema values and prints the Sionna mapping.

## Run Tests

```bash
python -m pytest -q
```
