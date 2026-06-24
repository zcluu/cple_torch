# Sionna Data

CPLE can generate CSI windows from a Sionna SYS-backed adapter.

## Scenario Profiles

Scenario YAML files live under `configs/`:

- `sionna_umi_low_mobility.yaml`
- `sionna_uma_medium_mobility.yaml`
- `sionna_rma_high_mobility.yaml`
- `sionna_inh_hotspot.yaml`

UMi, UMa, and RMa use Sionna TR 38.901 channel models when available. The InH
profile currently falls back to topology-aware generated tensors because no
3GPP InH channel class is wired into the adapter.

## CSIWindow

Each scheduled UE receives a `CSIWindow`:

```text
history: [history_len, *frame_shape]
current: [*frame_shape]
target:  [horizon + 1, *frame_shape]
```

`current` is the frame available at the service slot. `target` is the reference
window used by experiments.

## Shape Mapping

Sionna channel frequency responses are flattened and reshaped to the configured
`frame_shape`. If there are fewer generated values than requested, the adapter
repeats values before truncating to the configured frame size.

## Feedback Resources

The adapter uses Sionna's proportional-fair scheduler to derive available
resource counts per UE. CPLE consumes those resources when modeling feedback
scheduling.
