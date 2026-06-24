# API Usage

## Shape

```python
from cple import CSIShapeSpec

shape = CSIShapeSpec(
    frame_shape=(2, 4, 4),
    axes=("complex", "rx_ant", "subcarrier"),
    history_len=4,
    horizon=3,
    dtype="float32",
)
```

`horizon=3` means the final result has `horizon + 1 = 4` frames.

## fb-pred

```python
from cple import SerialNetwork

network = SerialNetwork.fb_pred(
    name="my_fb_pred",
    encoder=ue_encoder,
    bs_steps=[
        ("decoder", bs_decoder),
        ("predictor", bs_predictor),
        ("assemble", assemble_result),
    ],
    feedback_frames=1,
    prediction_frames=3,
)
```

Expected semantics:

```text
UE encoder: CSIWindow -> one feedback codeword
BS decoder: BSInput -> reconstructed current frame
BS predictor: reconstructed current + history -> P future frames
Final result: P+1 frames
```

## pred-fb

```python
network = SerialNetwork.pred_fb(
    name="my_pred_fb",
    predictor=ue_predictor,
    encoder=ue_encoder,
    bs_steps=[("decoder", bs_decoder)],
    prediction_frames=3,
)
```

The helper automatically feeds `current + P predicted future frames` into the
encoder one frame at a time. The default feedback frame count is `P+1`.

For advanced users, `ue_steps` may be provided directly. In that case, the step
that should receive each `pred-fb` frame must use:

```python
ModelStep("encoder", encoder, repeat="feedback_frames", input_source="pred_fb_frame")
```

## parallel

```python
from cple import ParallelNetwork

network = ParallelNetwork(
    name="my_parallel",
    encoder=ue_encoder,
    bs_network=bs_whole_network,
    feedback_frames=1,
    prediction_frames=3,
)
```

The framework does not split the BS network. The BS module may perform
decode-then-predict, predict-then-decode, codeword prediction, CSI prediction,
or any project-specific fusion logic.

## Running A Network

```python
from cple.configs.runner import build_adapter, load_config
from cple.runtime.platform import CPLEPlatform

config = load_config("configs/smoke.yaml")
adapter = build_adapter(config)
platform = CPLEPlatform(
    config=config.platform,
    shape=config.shape.to_spec(),
    feedback=config.feedback,
    adapter=adapter,
)

summaries = platform.run(network)
```

`summaries` contains:

- `latency_summary`
- `stage_summary`
