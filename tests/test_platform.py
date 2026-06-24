from pathlib import Path

import pytest

from cple.configs.runner import build_adapter, load_config, run_experiment
from cple.data.scenario import (
    describe_sionna_mapping,
    load_sionna_scenario,
    scenario_to_adapter_config,
    scenario_to_feedback_config,
    scenario_to_platform_config,
    scenario_to_shape_config,
)
from cple.data.sionna_env import inspect_sionna_environment


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_sionna_imports_environment():
    info = inspect_sionna_environment()
    assert info.available
    assert info.version is not None


def test_reference_sionna_scenario_profiles_load():
    for name in [
        "sionna_umi_low_mobility.yaml",
        "sionna_uma_medium_mobility.yaml",
        "sionna_inh_hotspot.yaml",
        "sionna_rma_high_mobility.yaml",
    ]:
        scenario = load_sionna_scenario(CONFIGS / name)
        platform = scenario_to_platform_config(scenario)
        adapter = scenario_to_adapter_config(scenario)
        shape = scenario_to_shape_config(scenario)
        feedback = scenario_to_feedback_config(scenario)
        mapping = describe_sionna_mapping(scenario)
        assert platform.num_slots > 0
        assert adapter.num_ues == scenario.ue.num_ues
        assert shape.to_spec().output_frames == shape.horizon + 1
        assert feedback.bits_per_resource_unit > 0
        assert mapping["scheduler"]["class"] == "sionna.sys.scheduling.PFSchedulerSUMIMO"


def test_smoke_config_uses_flow_schema():
    config = load_config(CONFIGS / "smoke.yaml")
    assert config.flow == "fb-pred"
    assert config.shape.frame_shape == (2, 4, 4)
    assert config.shape.history_len == 4
    assert config.shape.horizon == 3
    assert config.feedback.bitwidth == 8
    assert config.feedback.bits_per_resource_unit == 64


def test_config_validation_rejects_bad_values(tmp_path):
    bad_flow = tmp_path / "bad_flow.yaml"
    bad_flow.write_text("flow: invalid\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad_flow)

    bad_shape = tmp_path / "bad_shape.yaml"
    bad_shape.write_text(
        """
shape:
  frame_shape: [2, 0]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(bad_shape)

    bad_adapter = tmp_path / "bad_adapter.yaml"
    bad_adapter.write_text(
        """
adapter:
  num_ues: 1
  scheduled_per_slot: 2
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(bad_adapter)

    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("unknown: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(unknown)


def test_sionna_sys_adapter_produces_cfr_windows():
    pytest.importorskip("sionna")
    config = load_config(CONFIGS / "smoke.yaml")
    adapter = build_adapter(config)
    step = adapter.step(0)
    assert step.metadata["adapter"] == "sionna_sys"
    assert step.metadata["channel_source"] == "3gpp_tr38901_cfr"
    assert step.scheduled_ues
    ue_id = step.scheduled_ues[0]
    window = step.windows[ue_id]
    shape = config.shape.to_spec()
    assert tuple(window.history.shape) == (shape.history_len, *shape.frame_shape)
    assert tuple(window.current.shape) == shape.frame_shape
    assert tuple(window.target.shape) == (shape.output_frames, *shape.frame_shape)


def test_smoke_experiment_outputs(tmp_path):
    pd = pytest.importorskip("pandas")
    scenario_profile = tmp_path / "sionna_umi_low_mobility.yaml"
    scenario_profile.write_text(
        (CONFIGS / "sionna_umi_low_mobility.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        f"""
sionna_scenario_path: sionna_umi_low_mobility.yaml
platform:
  run_id: pytest_smoke
  num_slots: 4
  warmup_slots: 1
  output_dir: {output_dir.as_posix()}
  seed: 11
shape:
  frame_shape: [2, 4, 4]
  axes: [complex, rx_ant, subcarrier]
  history_len: 4
  horizon: 3
  dtype: float32
feedback:
  bitwidth: 8
  bits_per_resource_unit: 64
adapter:
  scheduled_per_slot: 2
  seed: 11
flow: parallel
""",
        encoding="utf-8",
    )
    outputs = run_experiment(config_path)
    assert set(outputs) == {"event_log", "latency_summary", "stage_summary", "environment"}
    latency = pd.read_csv(outputs["latency_summary"])
    stage = pd.read_csv(outputs["stage_summary"])
    event_log = pd.read_csv(outputs["event_log"])
    assert set(latency["flow_name"]) == {"dummy_parallel"}
    assert set(latency["flow_kind"]) == {"parallel"}
    assert {"mean_model_runtime_ms", "mean_scheduling_delay_ms", "mean_feedback_duration_ms"} <= set(latency.columns)
    assert {"ue_model", "air_feedback", "bs_model"} <= set(stage["stage_kind"])
    assert {"encoder", "air_feedback", "bs_network"} <= set(stage["stage_name"])
    assert {"flow_start", "stage_end", "flow_end"} <= set(event_log["event_type"])


def test_user_component_example_builds_one_flow():
    import sys

    examples_dir = (ROOT / "examples").resolve()
    sys.path.insert(0, str(examples_dir))
    try:
        from user_csi_models import build_user_flow
    finally:
        sys.path.remove(str(examples_dir))

    config = load_config(CONFIGS / "user_models.yaml")
    network = build_user_flow(shape=config.shape.to_spec(), device=config.platform.device, flow=config.flow)
    flow = network.build_flow(config.shape.to_spec())
    assert flow.flow.value == "parallel"
    assert flow.name == "user_parallel"
