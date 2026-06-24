from __future__ import annotations

import ast

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
        flows = df[df["event_type"] == EventType.FLOW_END]
        if flows.empty:
            return pd.DataFrame()
        rows = []
        for (flow_name, flow_kind), group in flows.groupby(["flow_name", "flow_kind"], dropna=False):
            latency = group["total_latency_ms"].astype(float)
            deadline = group["deadline_ms"].astype(float)
            stages = df[
                (df["event_type"] == EventType.STAGE_END)
                & (df["flow_name"] == flow_name)
                & (df["flow_kind"] == flow_kind)
            ]
            model_runtime = stages[stages["side"].isin(["ue", "bs"])]["runtime_ms"].astype(float)
            air_runtime = stages[stages["stage_kind"] == "air_feedback"]["runtime_ms"].astype(float)
            scheduling_delay = []
            feedback_duration = []
            for value in stages[stages["stage_kind"] == "air_feedback"]["metadata"]:
                metadata = self._parse_metadata(value)
                if "scheduling_delay_ms" in metadata:
                    scheduling_delay.append(float(metadata["scheduling_delay_ms"]))
                if "feedback_duration_ms" in metadata:
                    feedback_duration.append(float(metadata["feedback_duration_ms"]))
            rows.append(
                {
                    "flow_name": flow_name,
                    "flow_kind": flow_kind,
                    "service_count": int(len(group)),
                    "mean_model_runtime_ms": float(model_runtime.sum() / len(group)) if len(group) else 0.0,
                    "mean_air_feedback_ms": float(air_runtime.sum() / len(group)) if len(group) else 0.0,
                    "mean_scheduling_delay_ms": float(sum(scheduling_delay) / len(group)) if len(group) else 0.0,
                    "mean_feedback_duration_ms": float(sum(feedback_duration) / len(group)) if len(group) else 0.0,
                    "mean_total_latency_ms": float(latency.mean()),
                    "median_total_latency_ms": float(latency.median()),
                    "p95_total_latency_ms": float(latency.quantile(0.95)),
                    "p99_total_latency_ms": float(latency.quantile(0.99)),
                    "std_total_latency_ms": float(latency.std(ddof=0)),
                    "jitter_p95_p50_ms": float(latency.quantile(0.95) - latency.quantile(0.5)),
                    "deadline_miss_ratio": float((latency > deadline).mean()),
                }
            )
        return pd.DataFrame(rows)

    def compute_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        stages = df[df["event_type"] == EventType.STAGE_END]
        if stages.empty:
            return pd.DataFrame()
        rows = []
        for (flow_name, flow_kind, stage_name, stage_kind, side), group in stages.groupby(
            ["flow_name", "flow_kind", "stage_name", "stage_kind", "side"],
            dropna=False,
        ):
            runtime = group["runtime_ms"].astype(float)
            payload = group["payload_bits"].dropna().astype(float)
            scheduling_delay = []
            feedback_duration = []
            for value in group["metadata"]:
                metadata = self._parse_metadata(value)
                if "scheduling_delay_ms" in metadata:
                    scheduling_delay.append(float(metadata["scheduling_delay_ms"]))
                if "feedback_duration_ms" in metadata:
                    feedback_duration.append(float(metadata["feedback_duration_ms"]))
            rows.append(
                {
                    "flow_name": flow_name,
                    "flow_kind": flow_kind,
                    "stage_name": stage_name,
                    "stage_kind": stage_kind,
                    "side": side,
                    "event_count": int(len(group)),
                    "mean_stage_ms": float(runtime.mean()),
                    "p95_stage_ms": float(runtime.quantile(0.95)),
                    "p99_stage_ms": float(runtime.quantile(0.99)),
                    "mean_payload_bits": float(payload.mean()) if not payload.empty else 0.0,
                    "mean_scheduling_delay_ms": self._mean(scheduling_delay),
                    "mean_feedback_duration_ms": self._mean(feedback_duration),
                }
            )
        return pd.DataFrame(rows)

    def _parse_metadata(self, value) -> dict:
        if isinstance(value, dict):
            return value
        if not value:
            return {}
        try:
            parsed = ast.literal_eval(str(value))
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _mean(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))
