# Latency Model

CPLE separates latency into model runtime and feedback-channel timing.

```text
total latency = UE model runtime
              + air feedback scheduling delay
              + air feedback occupation time
              + BS model runtime
```

## Model Runtime

Model runtime is measured by `OnlineRuntimeProfiler`.

- CPU uses wall-clock timing with `time.perf_counter`.
- CUDA uses CUDA events and synchronizes around the measured callable.

Each `ModelStep` is profiled independently. Repeated steps, such as the
`pred-fb` encoder over `P+1` frames, are recorded as separate stage events.

## Scheduling Request Time

Feedback scheduling is anchored to the service slot time, not the end of UE
model execution.

This matters for fairness:

```text
fb-pred and parallel both feed back one current codeword.
If payload size is the same, their scheduling allocation should be the same.
```

UE runtime is still counted in total latency, but it does not change the
feedback resource allocation outcome for otherwise identical feedback packets.

## Feedback Payload Size

Payload bits are determined in this order:

1. If UE output has a `payload_bits` field or attribute, use it.
2. If UE output is a tensor, use `numel * feedback.bitwidth`.
3. If UE output is a list or tuple, sum the payload of each item.
4. Otherwise, estimate `elements_per_frame * feedback_frames * bitwidth`.

## Current Feedback Duration Rule

The Sionna SYS adapter currently estimates feedback duration as a fraction of a
slot:

```text
resource_units = ceil(payload_bits / bits_per_resource_unit)
feedback_duration = used_resource_units / available_resource_units * TTI
```

This is a resource-occupation approximation. If an experiment requires a fixed
per-codeword air time, the adapter should be extended with an explicit
`feedback_duration_ms` rule.

## Scheduling Delay

The adapter searches slots until the UE has enough remaining feedback resources.
If resources are available in the request slot, scheduling delay is zero. If
not, delay accumulates until a later slot can serve the feedback.
