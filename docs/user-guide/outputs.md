# Outputs And Metrics

CPLE writes CSV outputs through `CPLEPlatform.export_outputs`.

## event_log.csv

Raw event stream. Event types include:

- `slot_start`
- `slot_end`
- `flow_start`
- `stage_end`
- `flow_end`

Important columns:

| Column | Meaning |
| --- | --- |
| `slot_idx` | Service slot index |
| `ue_id` | UE identifier |
| `flow_name` | Network name |
| `flow_kind` | `fb-pred`, `pred-fb`, or `parallel` |
| `stage_name` | Model step or air stage name |
| `stage_kind` | `ue_model`, `air_feedback`, or `bs_model` |
| `runtime_ms` | Stage runtime |
| `payload_bits` | Feedback payload for air stages |
| `metadata` | Serialized metadata dictionary |

## latency_summary.csv

Aggregated per flow:

- service count
- model runtime
- air feedback runtime
- scheduling delay
- feedback duration
- mean, median, p95, p99 latency
- jitter
- deadline miss ratio

## stage_summary.csv

Aggregated by flow and stage:

- event count
- mean stage runtime
- p95 and p99 stage runtime
- mean payload bits
- mean scheduling delay
- mean feedback duration

## Warmup

Events from slots where `slot_idx < warmup_slots` are retained in the raw log
but excluded from summary metrics.
