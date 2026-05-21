from __future__ import annotations

import pandas as pd

from .events import CPLEEvent, EventType


class CPLEMetricsEngine:
    def to_frame(self, events: list[CPLEEvent]) -> pd.DataFrame:
        return pd.DataFrame([event.to_dict() for event in events])

    def compute_all(self, events: list[CPLEEvent]) -> dict[str, pd.DataFrame]:
        df = self.to_frame(events)
        return {
            "latency_summary": self.compute_latency(df),
            "stage_summary": self.compute_stage(df),
        }

    def compute_latency(self, df: pd.DataFrame) -> pd.DataFrame:
        deliveries = df[df["event_type"] == EventType.DELIVERY]
        if deliveries.empty:
            return pd.DataFrame()
        rows = []
        for (model_name, mode), group in deliveries.groupby(["model_name", "mode"], dropna=False):
            model_latency = group["runtime_ms"].astype(float)
            scheduling_delay = group["scheduling_delay_ms"].fillna(0).astype(float)
            feedback_duration = group["feedback_duration_ms"].fillna(0).astype(float)
            total_latency = group["total_latency_ms"].fillna(
                model_latency + scheduling_delay + feedback_duration
            ).astype(float)
            deadline = group["deadline_ms"].astype(float)
            outputs = group["output_frames"].apply(lambda value: len(str(value).split(",")) if value else 0)
            rows.append(
                {
                    "model_name": model_name,
                    "mode": mode,
                    "service_count": int(len(group)),
                    "mean_model_latency_ms": float(model_latency.mean()),
                    "mean_scheduling_delay_ms": float(scheduling_delay.mean()),
                    "mean_feedback_duration_ms": float(feedback_duration.mean()),
                    "mean_total_latency_ms": float(total_latency.mean()),
                    "median_total_latency_ms": float(total_latency.median()),
                    "p95_total_latency_ms": float(total_latency.quantile(0.95)),
                    "p99_total_latency_ms": float(total_latency.quantile(0.99)),
                    "std_total_latency_ms": float(total_latency.std(ddof=0)),
                    "jitter_p95_p50_ms": float(total_latency.quantile(0.95) - total_latency.quantile(0.5)),
                    "deadline_miss_ratio": float((total_latency > deadline).mean()),
                    "mean_outputs_per_call": float(outputs.mean()),
                    "mean_total_latency_per_output_ms": float((total_latency / outputs.replace(0, 1)).mean()),
                }
            )
        return pd.DataFrame(rows)

    def compute_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        stages = df[df["event_type"] == EventType.STAGE_END]
        if stages.empty:
            return pd.DataFrame()
        rows = []
        for (model_name, stage_name, operation_type), group in stages.groupby(["model_name", "stage_name", "operation_type"], dropna=False):
            runtime = group["runtime_ms"].astype(float)
            rows.append(
                {
                    "model_name": model_name,
                    "stage_name": stage_name,
                    "operation_type": operation_type,
                    "service_count": int(len(group)),
                    "mean_stage_ms": float(runtime.mean()),
                    "p95_stage_ms": float(runtime.quantile(0.95)),
                    "p99_stage_ms": float(runtime.quantile(0.99)),
                }
            )
        return pd.DataFrame(rows)
