from __future__ import annotations

import platform as py_platform
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from .adapters import MockSionnaAdapter
from .config import ExperimentConfig, MockAdapterConfig, PlatformConfig
from .models import DummyParallelModel, DummySerialModel
from .platform import CPLEPlatform
from .scenario import load_sionna_scenario, scenario_to_mock_adapter_config, scenario_to_platform_config
from .sionna_adapter import inspect_sionna_environment


def _merge_dataclass(cls, data: dict[str, Any]):
    base = cls()
    for key, value in data.items():
        setattr(base, key, value)
    return base


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    scenario_path = data.get("sionna_scenario_path")
    if scenario_path:
        scenario_file = Path(scenario_path)
        if not scenario_file.is_absolute():
            scenario_file = Path(path).parent / scenario_file
        scenario = load_sionna_scenario(scenario_file)
        platform_config = scenario_to_platform_config(scenario)
        adapter_config = scenario_to_mock_adapter_config(scenario)
        platform_config = _merge_dataclass(lambda: platform_config, data.get("platform", {}))
        adapter_config = _merge_dataclass(lambda: adapter_config, data.get("adapter", {}))
    else:
        platform_config = _merge_dataclass(PlatformConfig, data.get("platform", {}))
        adapter_config = _merge_dataclass(MockAdapterConfig, data.get("adapter", {}))
    return ExperimentConfig(
        platform=platform_config,
        adapter=adapter_config,
        models=data.get("models", ["dummy_parallel", "dummy_serial"]),
        sionna_scenario_path=str(scenario_path) if scenario_path else None,
    )


def build_models(config: ExperimentConfig):
    models = []
    for name in config.models:
        if name == "dummy_parallel":
            models.append(DummyParallelModel(csi_dim=config.adapter.csi_dim))
        elif name == "dummy_serial":
            models.append(DummySerialModel(csi_dim=config.adapter.csi_dim))
        else:
            raise ValueError(f"Unknown model: {name}")
    return models


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
    config = load_config(config_path)
    torch.manual_seed(config.platform.seed)
    adapter = MockSionnaAdapter(config.adapter, tti_ms=config.platform.tti_ms)
    adapter.reset(config.platform.seed)
    platform = CPLEPlatform(config.platform, adapter, build_models(config))
    platform.run()
    output_dir = Path(config.platform.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        (output_dir / "config.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
    paths = platform.export_outputs(output_dir)
    write_environment(output_dir / "environment.txt")
    paths["environment"] = output_dir / "environment.txt"
    return paths
