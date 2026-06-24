# CPLE Flow I/O Contract

This document defines the intended module boundaries, inputs, outputs, and
pipelines for `fb-pred`, `pred-fb`, and `parallel`.

CPLE is a latency evaluation tool. It measures model runtime plus feedback
scheduling delay. It does not define prediction accuracy metrics.

## Common Definitions

Let:

- `P`: prediction horizon, configured by the user.
- `R = P + 1`: final result frame count.
- `current`: the CSI frame available at the current slot.
- `history`: previous CSI frames available to the model.
- `target`: the reference result window used by experiments, with `R` frames.

Example:

```text
P = 3
R = P + 1 = 4

final result = 1 feedback/reconstructed frame + 3 predicted future frames
```

The most important distinction is:

```text
final result frames != air feedback frames
```

All three strategies should produce a final result with `R` frames. They differ
in how many frames are actually sent through feedback scheduling.

## Shared Runtime Objects

### CSIWindow

Input object available before a strategy starts.

```text
CSIWindow.history: [history_len, *frame_shape]
CSIWindow.current: [*frame_shape]
CSIWindow.target:  [R, *frame_shape]
```

### BSInput

Input object passed to the BS-side module or step after air feedback finishes.

```text
BSInput.window          = original CSIWindow
BSInput.ue_output       = output from UE-side model path
BSInput.flow            = fb-pred | pred-fb | parallel
BSInput.feedback_frames = number of air feedback frames scheduled
```

`BSInput.feedback_frames` is a scheduling quantity. It is not the final result
frame count.

## fb-pred

`fb-pred` means feedback first, prediction second.

### Intent

The UE only encodes one current CSI frame. The BS receives that feedback,
reconstructs the current frame, and predicts the next `P` future frames.

For `P = 3`:

```text
air feedback frames = 1
final result frames = 4
```

### Modules

#### UE encoder

Input:

```text
CSIWindow
```

Expected logical content used:

```text
CSIWindow.current
```

Output:

```text
compressed feedback for 1 current frame
```

The output can be a tensor or a user-defined object. If it contains
`payload_bits`, CPLE uses that as the feedback payload size. Otherwise CPLE
estimates payload bits from tensor size and feedback bitwidth.

#### BS decoder

Input:

```text
BSInput.ue_output
```

Output:

```text
reconstructed current frame: [*frame_shape]
```

This is one frame, not `P` frames and not `R` frames.

#### BS predictor

Input:

```text
history plus reconstructed current frame
```

Output:

```text
predicted future frames: [P, *frame_shape]
```

#### Final assembly

Logical final result:

```text
concat(reconstructed_current, predicted_future)
shape = [R, *frame_shape]
```

### Pipeline

```text
UE:
  CSIWindow.current
    -> encoder
    -> compressed 1-frame feedback

AIR:
  schedule/send 1 feedback frame

BS:
  compressed 1-frame feedback
    -> decoder
    -> reconstructed current frame
    -> predictor with history
    -> P predicted future frames
    -> concat
    -> R-frame final result
```

## pred-fb

`pred-fb` means prediction first, feedback second.

### Intent

The UE predicts future frames first. Then the UE encodes the current frame plus
the predicted future frames and sends all `R = P + 1` frames through feedback.
The BS reconstructs the complete result window.

For `P = 3`:

```text
air feedback frames = 4
final result frames = 4
```

This is the strategy expected to have higher scheduling pressure, because it
uses multiple feedback frames.

### Modules

#### UE predictor

Input:

```text
CSIWindow.history and/or CSIWindow.current
```

Output:

```text
predicted future frames: [P, *frame_shape]
```

#### UE result formation

Logical UE-side result to be fed back:

```text
concat(current, predicted_future)
shape = [R, *frame_shape]
```

This formation can be explicit in the user's code or internal to the user's UE
model path. CPLE does not need to own this tensor construction unless the user
chooses to expose separate predictor and encoder steps.

#### UE encoder

Input:

```text
one frame from the R-frame UE-side result
```

or, if the user's encoder is batch-oriented:

```text
the full R-frame UE-side result
```

Expected output:

```text
compressed feedback for R frames
```

For latency accounting, CPLE should model this as `R` feedback-frame payloads.
If the encoder is measured per frame, the encoder runtime is repeated `R`
times. If the user provides a batched encoder, the framework must still account
for `R` air feedback frames.

#### BS decoder

Input:

```text
compressed feedback for R frames
```

Output:

```text
reconstructed result frames: [R, *frame_shape]
```

### Pipeline

```text
UE:
  CSIWindow
    -> predictor
    -> P predicted future frames
    -> concat current + predicted future
    -> R frames
    -> encoder
    -> compressed R-frame feedback

AIR:
  schedule/send R feedback frames

BS:
  compressed R-frame feedback
    -> decoder
    -> R-frame final result
```

## parallel

`parallel` means the framework only separates the UE part and the BS part. The
BS-side internal order is user-defined.

### Intent

The UE encodes one current CSI frame. The BS receives that one feedback frame
and runs a user-provided whole BS network to produce the final `R` frames.

For `P = 3`:

```text
air feedback frames = 1
final result frames = 4
```

The framework must not force the BS network to be `decoder -> predictor` or
`predictor -> decoder`. Both are valid project designs.

### Modules

#### UE encoder

Input:

```text
CSIWindow
```

Expected logical content used:

```text
CSIWindow.current
```

Output:

```text
compressed feedback for 1 current frame
```

#### BS whole network

Input:

```text
BSInput
```

Output:

```text
final result frames: [R, *frame_shape]
```

The BS whole network may internally do any of the following:

- decode then predict
- predict then decode
- predict codewords
- predict CSI
- refine decoded or predicted frames
- use any project-specific fusion logic

CPLE treats it as one BS-side callable for runtime measurement.

### Pipeline

```text
UE:
  CSIWindow.current
    -> encoder
    -> compressed 1-frame feedback

AIR:
  schedule/send 1 feedback frame

BS:
  BSInput(window, ue_output, flow=parallel, feedback_frames=1)
    -> user BS whole network
    -> R-frame final result
```

## Strategy Comparison Summary

For prediction horizon `P`:

| Strategy | UE-side work | Air feedback frames | BS-side work | Final result frames |
| --- | --- | ---: | --- | ---: |
| `fb-pred` | encode current frame | 1 | decode 1 frame, predict `P` frames, assemble | `P + 1` |
| `pred-fb` | predict `P` frames, encode `P + 1` frames | `P + 1` | decode `P + 1` frames | `P + 1` |
| `parallel` | encode current frame | 1 | user-defined whole BS network | `P + 1` |

## Implementation Notes

- `prediction_frames` should mean `P`.
- `result_frames` should mean `P + 1`.
- `feedback_frames` or `air_feedback_frames` should mean frames scheduled over
  the feedback channel.
- Feedback scheduling requests are anchored to the service slot time, not to
  the end of UE model execution. UE model runtime is still counted in total
  latency, but it must not change the resource allocation outcome for two flows
  that send the same payload at the same service slot.
- Avoid using `feedback_frames` to mean final output frames.
- For `fb-pred`, the clean semantic contract is decoder output `1` frame and
  predictor output `P` frames, then assemble `P + 1` frames.
- For `parallel`, the framework must not inspect or split the BS network.
- For `pred-fb`, scheduling should account for `P + 1` feedback frames.
- The `SerialNetwork.pred_fb(predictor=..., encoder=...)` helper automatically
  feeds `current + P predicted frames` into the encoder one frame at a time. If
  users build custom `ue_steps`, the encoder step must opt into that same
  per-frame input behavior.
