from pathlib import Path

import pandas as pd

from cple.runner import run_experiment
from cple.scenario import describe_sionna_mapping, load_sionna_scenario, scenario_to_mock_adapter_config, scenario_to_platform_config
from cple.sionna_adapter import inspect_sionna_environment


def test_sionna_imports_cpu_environment():
    info = inspect_sionna_environment()
    assert info.available
    assert info.version == "2.0.1"


def test_smoke_experiment_outputs(tmp_path):
    config_path = tmp_path / "smoke.yaml"
    output_dir = tmp_path / "outputs"
    config_path.write_text(
        f"""
platform:
  run_id: pytest_smoke
  num_slots: 8
  tti_ms: 5.0
  deadline_ms: 5.0
  device: cpu
  output_dir: {output_dir.as_posix()}
  seed: 11
adapter:
  num_ues: 4
  csi_dim: 8
  scheduled_per_slot: 2
  history_len: 2
  seed: 11
models:
  - dummy_parallel
  - dummy_serial
""",
        encoding="utf-8",
    )
    outputs = run_experiment(config_path)
    latency = pd.read_csv(outputs["latency_summary"])
    stage = pd.read_csv(outputs["stage_summary"])
    assert {"dummy_parallel", "dummy_serial"} <= set(latency["model_name"])
    assert "mean_total_latency_ms" in latency.columns
    assert "mean_scheduling_delay_ms" in latency.columns
    assert not stage.empty
    assert set(outputs) == {"latency_summary", "stage_summary", "environment"}


def test_reference_sionna_scenario_profiles_load():
    for name in [
        "sionna_umi_low_mobility.yaml",
        "sionna_uma_medium_mobility.yaml",
        "sionna_inh_hotspot.yaml",
        "sionna_rma_high_mobility.yaml",
    ]:
        config = load_sionna_scenario(Path("configs") / name)
        platform = scenario_to_platform_config(config)
        adapter = scenario_to_mock_adapter_config(config)
        mapping = describe_sionna_mapping(config)
        assert platform.num_slots > 0
        assert adapter.num_ues == config.ue.num_ues
        assert mapping["scheduler"]["num_ut"] == config.ue.num_ues


def test_scenario_smoke_config_outputs(tmp_path):
    scenario_source = Path("configs/scenario_smoke.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "scenario_smoke.yaml"
    scenario_profile = tmp_path / "sionna_umi_low_mobility.yaml"
    scenario_profile.write_text(Path("configs/sionna_umi_low_mobility.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    config_path.write_text(
        scenario_source.replace("outputs/scenario_smoke", (tmp_path / "scenario_outputs").as_posix()),
        encoding="utf-8",
    )
    outputs = run_experiment(config_path)
    latency = pd.read_csv(outputs["latency_summary"])
    assert not latency.empty
    assert set(outputs) == {"latency_summary", "stage_summary", "environment"}


def test_user_model_example_runs(tmp_path):
    import sys

    examples_dir = Path("examples").resolve()
    sys.path.insert(0, str(examples_dir))
    try:
        from user_csi_models import build_user_models
    finally:
        sys.path.remove(str(examples_dir))

    from cple import CPLEPlatform
    from cple.adapters import MockSionnaAdapter
    from cple.runner import load_config

    scenario_profile = tmp_path / "sionna_umi_low_mobility.yaml"
    scenario_profile.write_text(Path("configs/sionna_umi_low_mobility.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    config_path = tmp_path / "user_models.yaml"
    config_path.write_text(
        f"""
sionna_scenario_path: sionna_umi_low_mobility.yaml
platform:
  run_id: pytest_user_models
  num_slots: 6
  output_dir: {(tmp_path / "user_models").as_posix()}
adapter:
  csi_dim: 16
  scheduled_per_slot: 2
models:
  - user_serial_predict_then_feedback
  - user_parallel_feedback_then_predict
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    platform = CPLEPlatform(
        config.platform,
        MockSionnaAdapter(config.adapter, tti_ms=config.platform.tti_ms),
        build_user_models(csi_dim=config.adapter.csi_dim, horizon=3),
    )
    platform.run()
    outputs = platform.export_outputs(config.platform.output_dir)
    latency = pd.read_csv(outputs["latency_summary"])
    stage = pd.read_csv(outputs["stage_summary"])
    assert {"user_serial_predict_then_feedback", "user_parallel_feedback_then_predict"} <= set(latency["model_name"])
    assert {"predict_future_p_frames", "feedback_frame_0", "feedback_frame_1", "feedback_frame_2", "feedback_frame_3"} <= set(stage["stage_name"])
    by_model = latency.set_index("model_name")
    assert by_model.loc["user_serial_predict_then_feedback", "mean_feedback_duration_ms"] > by_model.loc["user_parallel_feedback_then_predict", "mean_feedback_duration_ms"]
    assert by_model.loc["user_serial_predict_then_feedback", "mean_total_latency_ms"] > by_model.loc["user_parallel_feedback_then_predict", "mean_total_latency_ms"]
    assert set(outputs) == {"latency_summary", "stage_summary"}
