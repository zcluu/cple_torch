# CPLE Torch

CPLE Torch is a lightweight latency evaluation tool for CSI feedback and prediction
pipelines. It is designed for experiments where PyTorch model runtime alone is
not enough because feedback scheduling delay must also be modeled.

CPLE Torch measures:

```text
UE model runtime -> air feedback scheduling/occupation -> BS model runtime
```

It does not evaluate prediction quality. Accuracy metrics such as NMSE, SGCS,
and RMSE belong to the model project or dataset pipeline.

## Main Use Case

CPLE Torch compares three strategy families:

| Strategy | UE side | Feedback channel | BS side |
| --- | --- | --- | --- |
| `fb-pred` | encode current CSI/codeword | one feedback frame | decode one frame, predict future frames |
| `pred-fb` | predict future CSI, encode `P+1` frames | `P+1` feedback frames | decode `P+1` frames |
| `parallel` | encode current CSI/codeword | one feedback frame | user-provided whole BS network |

For prediction horizon `P=3`, all strategies target `P+1=4` final frames.
They differ in how much they feed back over the air.

## Package Layout

```text
cple/api.py          public flow, shape, and result types
cple/runtime/       executor, platform, profiler
cple/data/          Sionna scenario and adapter implementation
cple/configs/       YAML schema and runner helpers
cple/reporting/     event log and metrics
cple/models/        dummy and benchmark model builders
examples/           runnable integrations and comparison scripts
configs/            smoke and Sionna scenario YAML files
```

## Documentation Map

- [Flow I/O Contract](concepts/flow-io.md): exact module inputs, outputs, and
  pipelines.
- [API Usage](user-guide/api-usage.md): how to construct and run networks.
- [Configuration](user-guide/configuration.md): YAML fields and scenario
  mapping.
- [API Reference](api/public.md): generated reference for all public and
  internal APIs.
