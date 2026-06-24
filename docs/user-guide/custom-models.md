# Custom Models

CPLE accepts ordinary callables, including `torch.nn.Module` instances.

## Callable Inputs

UE-side modules receive a `CSIWindow` unless they are repeated as `pred-fb`
per-frame encoders. BS-side modules receive `BSInput`.

```python
class MyEncoder(torch.nn.Module):
    def forward(self, window: CSIWindow):
        return encode(window.current)

class MyBSNetwork(torch.nn.Module):
    def forward(self, data: BSInput):
        return decode_or_predict(data.window, data.ue_output)
```

## Payload Bits Override

If the UE output has a `payload_bits` field or attribute, CPLE uses it directly:

```python
def forward(self, window):
    return {
        "latent": latent,
        "payload_bits": 256,
    }
```

Without an override, tensor payload is estimated as:

```text
tensor.numel() * feedback.bitwidth
```

## Device Handling

CPLE passes `context.device` into the profiler. The framework does not
automatically move arbitrary user tensors to a model device. Custom wrappers
should move `CSIWindow` or `BSInput` tensors as needed.

## Result Shapes

For `horizon=P`, final result tensors should normally have:

```text
[P + 1, *frame_shape]
```

CPLE is primarily a latency tool, so it does not score result accuracy.
