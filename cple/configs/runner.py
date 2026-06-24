from __future__ import annotations

import platform as py_platform
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from .schema import (
    AdapterConfig,
    ExperimentConfig,
    FeedbackConfig,
    PlatformConfig,
    ShapeConfig,
    dataclass_field_names,
    merge_dataclass,
)
from ..data.scenario import (
    load_sionna_scenario,
    scenario_to_adapter_config,
    scenario_to_feedback_config,
    scenario_to_platform_config,
    scenario_to_shape_config,
)
from ..data.sionna_env import inspect_sionna_environment
from ..data.sionna_sys import SionnaSysAdapter
from ..models import build_dummy_flow


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _validate_top_level(data)

    scenario_path = data.get("sionna_scenario_path")
    if scenario_path:
        scenario_file = Path(scenario_path)
        if not scenario_file.is_absolute():
            scenario_file = path.parent / scenario_file
        scenario = load_sionna_scenario(scenario_file)
        platform_config = scenario_to_platform_config(scenario)
        shape_config = scenario_to_shape_config(scenario)
        feedback_config = scenario_to_feedback_config(scenario)
        adapter_config = scenario_to_adapter_config(scenario)
        merge_dataclass(platform_config, data.get("platform", {}))
        merge_dataclass(shape_config, data.get("shape", {}))
        merge_dataclass(feedback_config, data.get("feedback", {}))
        merge_dataclass(adapter_config, data.get("adapter", {}))
    else:
        scenario_file = None
        platform_config = merge_dataclass(PlatformConfig, data.get("platform", {}))
        shape_config = merge_dataclass(ShapeConfig, data.get("shape", {}))
        feedback_config = merge_dataclass(FeedbackConfig, data.get("feedback", {}))
        adapter_config = merge_dataclass(AdapterConfig, data.get("adapter", {}))

    config = ExperimentConfig(
        platform=platform_config,
        shape=shape_config,
        feedback=feedback_config,
        adapter=adapter_config,
        flow=data.get("flow", ExperimentConfig().flow),
        sionna_scenario_path=str(scenario_file) if scenario_path else None,
    )
    config.validate()
    return config


def build_adapter(config: ExperimentConfig) -> SionnaSysAdapter:
    if config.sionna_scenario_path is None:
        raise ValueError("A Sionna scenario YAML is required for CPLE runtime")
    scenario = load_sionna_scenario(config.sionna_scenario_path)
    adapter = SionnaSysAdapter(
        scenario=scenario,
        adapter_config=config.adapter,
        shape=config.shape.to_spec(),
        feedback=config.feedback,
        tti_ms=config.platform.tti_ms,
        device=config.platform.device,
    )
    adapter.reset(config.platform.seed)
    return adapter


def build_flow(config: ExperimentConfig):
    return build_dummy_flow(
        shape=config.shape.to_spec(),
        device=config.platform.device,
        flow=config.flow,
    )


def write_environment(path: Path) -> None:
    info = inspect_sionna_environment()
    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
    path.write_text(
        "\n".join(
            [
                f"python={py_platform.python_version()}",
                f"torch={torch.__version__}",
                f"cuda_available={torch.cuda.is_available()}",
                f"sionna_available={info.available}",
                f"sionna_version={info.version}",
                f"sionna_note={info.note}",
                f"git_commit={git_commit}",
            ]
        ),
        encoding="utf-8",
    )


def run_experiment(config_path: str | Path | None = None) -> dict[str, Path]:
    from ..runtime.platform import CPLEPlatform

    config = load_config(config_path)
    torch.manual_seed(config.platform.seed)
    adapter = build_adapter(config)
    platform = CPLEPlatform(
        config=config.platform,
        shape=config.shape.to_spec(),
        feedback=config.feedback,
        adapter=adapter,
    )
    platform.run(build_flow(config))
    output_dir = Path(config.platform.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        (output_dir / "config.yaml").write_text(
            Path(config_path).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    paths = platform.export_outputs(output_dir)
    write_environment(output_dir / "environment.txt")
    paths["environment"] = output_dir / "environment.txt"
    return paths


def _validate_top_level(data: dict[str, Any]) -> None:
    allowed = dataclass_field_names(ExperimentConfig)
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown ExperimentConfig fields: {sorted(unknown)}")
