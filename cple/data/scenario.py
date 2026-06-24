from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from ..configs.schema import AdapterConfig, FeedbackConfig, PlatformConfig, ShapeConfig


ScenarioName = Literal["umi", "uma", "rma", "inh"]
DuplexMode = Literal["uplink", "downlink"]
SchedulerType = Literal["proportional_fair"]


@dataclass
class ScenarioTimeConfig:
    num_slots: int = 100
    tti_ms: float = 1.0
    warmup_slots: int = 5


@dataclass
class ScenarioNRConfig:
    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    subcarrier_spacing_hz: float = 30e3
    num_ofdm_symbols: int = 14
    num_frequency_resources: int = 52
    duplex_mode: DuplexMode = "downlink"


@dataclass
class ScenarioTopologyConfig:
    scenario: ScenarioName = "umi"
    num_rings: int = 1
    num_cells: int = 3
    sectors_per_cell: int = 3
    inter_site_distance_m: float = 200.0
    cell_radius_m: float = 100.0
    min_bs_ut_distance_m: float = 10.0
    max_bs_ut_distance_m: float = 150.0
    bs_height_m: float = 10.0
    min_ut_height_m: float = 1.5
    max_ut_height_m: float = 1.5
    indoor_probability: float = 0.2


@dataclass
class ScenarioUEConfig:
    num_ues: int = 16
    scheduled_per_slot: int = 4
    min_velocity_mps: float = 0.0
    max_velocity_mps: float = 3.0
    traffic_model: str = "scheduled_only"


@dataclass
class ScenarioChannelConfig:
    model_family: str = "tr38901"
    model: ScenarioName = "umi"
    los: str | bool = "auto"
    normalize_channel: bool = True
    delay_spread_s: float | None = None
    use_3gpp_channel: bool = True
    o2i_model: str = "low"
    enable_pathloss: bool = True
    enable_shadow_fading: bool = True


@dataclass
class ScenarioSchedulerConfig:
    type: SchedulerType = "proportional_fair"
    num_streams_per_ue: int = 1
    beta: float = 0.98


@dataclass
class ScenarioLinkAdaptationConfig:
    enabled: bool = True
    olla_enabled: bool = True
    target_bler: float = 0.1


@dataclass
class ScenarioPowerControlConfig:
    uplink: str = "open_loop"
    downlink: str = "fair"
    max_tx_power_dbm: float = 23.0


@dataclass
class ScenarioCPLEConfig:
    frame_shape: tuple[int, ...] = (2, 8, 8)
    axes: tuple[str, ...] = ("complex", "rx_ant", "subcarrier")
    dtype: str = "float32"
    history_len: int = 4
    horizon: int = 3
    deadline_ms: float = 5.0
    feedback_bitwidth: int = 16
    bits_per_resource_unit: int = 64


@dataclass
class SionnaScenarioConfig:
    name: str = "urban_micro_low_mobility"
    description: str = ""
    seed: int = 7
    time: ScenarioTimeConfig = field(default_factory=ScenarioTimeConfig)
    nr: ScenarioNRConfig = field(default_factory=ScenarioNRConfig)
    topology: ScenarioTopologyConfig = field(default_factory=ScenarioTopologyConfig)
    ue: ScenarioUEConfig = field(default_factory=ScenarioUEConfig)
    channel: ScenarioChannelConfig = field(default_factory=ScenarioChannelConfig)
    scheduler: ScenarioSchedulerConfig = field(default_factory=ScenarioSchedulerConfig)
    link_adaptation: ScenarioLinkAdaptationConfig = field(default_factory=ScenarioLinkAdaptationConfig)
    power_control: ScenarioPowerControlConfig = field(default_factory=ScenarioPowerControlConfig)
    cple: ScenarioCPLEConfig = field(default_factory=ScenarioCPLEConfig)
    metadata: dict[str, Any] = field(default_factory=dict)


def _merge_dataclass(cls, data: dict[str, Any]):
    obj = cls()
    for key, value in (data or {}).items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown {cls.__name__} field: {key}")
        current = getattr(obj, key)
        if isinstance(current, float) and isinstance(value, str):
            value = float(value)
        elif isinstance(current, int) and isinstance(value, str):
            value = int(value)
        elif isinstance(current, bool) and isinstance(value, str):
            value = value.lower() in {"1", "true", "yes"}
        elif isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(obj, key, value)
    return obj


def load_sionna_scenario(path: str | Path) -> SionnaScenarioConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    scenario = SionnaScenarioConfig(
        name=data.get("name", path.stem),
        description=data.get("description", ""),
        seed=data.get("seed", 7),
        time=_merge_dataclass(ScenarioTimeConfig, data.get("time", {})),
        nr=_merge_dataclass(ScenarioNRConfig, data.get("nr", {})),
        topology=_merge_dataclass(ScenarioTopologyConfig, data.get("topology", {})),
        ue=_merge_dataclass(ScenarioUEConfig, data.get("ue", {})),
        channel=_merge_dataclass(ScenarioChannelConfig, data.get("channel", {})),
        scheduler=_merge_dataclass(ScenarioSchedulerConfig, data.get("scheduler", {})),
        link_adaptation=_merge_dataclass(ScenarioLinkAdaptationConfig, data.get("link_adaptation", {})),
        power_control=_merge_dataclass(ScenarioPowerControlConfig, data.get("power_control", {})),
        cple=_merge_dataclass(ScenarioCPLEConfig, data.get("cple", {})),
        metadata=data.get("metadata", {}),
    )
    validate_sionna_scenario(scenario)
    return scenario


def validate_sionna_scenario(config: SionnaScenarioConfig) -> None:
    allowed_scenarios = {"umi", "uma", "rma", "inh"}
    if config.topology.scenario not in allowed_scenarios:
        raise ValueError(f"topology.scenario must be one of {sorted(allowed_scenarios)}")
    if config.channel.model not in allowed_scenarios:
        raise ValueError(f"channel.model must be one of {sorted(allowed_scenarios)}")
    if config.channel.model != config.topology.scenario:
        raise ValueError("channel.model should match topology.scenario for TR 38.901 profiles")
    if config.channel.o2i_model not in {"low", "high"}:
        raise ValueError("channel.o2i_model must be low or high")
    if config.time.num_slots <= 0:
        raise ValueError("time.num_slots must be positive")
    if config.time.tti_ms <= 0:
        raise ValueError("time.tti_ms must be positive")
    if config.ue.num_ues <= 0:
        raise ValueError("ue.num_ues must be positive")
    if config.ue.scheduled_per_slot <= 0:
        raise ValueError("ue.scheduled_per_slot must be positive")
    if config.ue.scheduled_per_slot > config.ue.num_ues:
        raise ValueError("ue.scheduled_per_slot cannot exceed ue.num_ues")
    if not config.cple.frame_shape:
        raise ValueError("cple.frame_shape must not be empty")
    if any(dim <= 0 for dim in config.cple.frame_shape):
        raise ValueError("cple.frame_shape dimensions must be positive")
    if len(config.cple.frame_shape) != len(config.cple.axes):
        raise ValueError("cple.frame_shape and cple.axes must have the same length")
    if config.cple.dtype not in {"float32", "complex64"}:
        raise ValueError("cple.dtype must be float32 or complex64")
    if config.cple.history_len <= 0:
        raise ValueError("cple.history_len must be positive")
    if config.cple.horizon <= 0:
        raise ValueError("cple.horizon must be positive")
    if config.cple.feedback_bitwidth <= 0:
        raise ValueError("cple.feedback_bitwidth must be positive")
    if config.cple.bits_per_resource_unit <= 0:
        raise ValueError("cple.bits_per_resource_unit must be positive")
    if config.nr.carrier_frequency_hz <= 0:
        raise ValueError("nr.carrier_frequency_hz must be positive")
    if config.nr.bandwidth_hz <= 0:
        raise ValueError("nr.bandwidth_hz must be positive")
    if config.scheduler.type != "proportional_fair":
        raise ValueError("scheduler.type must be proportional_fair")


def scenario_to_adapter_config(config: SionnaScenarioConfig) -> AdapterConfig:
    return AdapterConfig(
        num_ues=config.ue.num_ues,
        scheduled_per_slot=config.ue.scheduled_per_slot,
        seed=config.seed,
    )


def scenario_to_platform_config(config: SionnaScenarioConfig, output_dir: str | None = None) -> PlatformConfig:
    return PlatformConfig(
        run_id=config.name,
        num_slots=config.time.num_slots,
        tti_ms=config.time.tti_ms,
        deadline_ms=config.cple.deadline_ms,
        device="cpu",
        warmup_slots=config.time.warmup_slots,
        output_dir=output_dir or f"outputs/{config.name}",
        seed=config.seed,
    )


def scenario_to_shape_config(config: SionnaScenarioConfig) -> ShapeConfig:
    return ShapeConfig(
        frame_shape=tuple(config.cple.frame_shape),
        axes=tuple(config.cple.axes),
        dtype=config.cple.dtype,
        history_len=config.cple.history_len,
        horizon=config.cple.horizon,
    )


def scenario_to_feedback_config(config: SionnaScenarioConfig) -> FeedbackConfig:
    return FeedbackConfig(
        bitwidth=config.cple.feedback_bitwidth,
        bits_per_resource_unit=config.cple.bits_per_resource_unit,
    )


def describe_sionna_mapping(config: SionnaScenarioConfig) -> dict[str, Any]:
    """Return the intended Sionna SYS mapping without instantiating a full scene."""
    return {
        "topology": {
            "function": "sionna.sys.topology.gen_hexgrid_topology",
            "scenario": config.topology.scenario,
            "num_rings": config.topology.num_rings,
            "isd": config.topology.inter_site_distance_m,
            "num_ut_per_sector": max(1, config.ue.num_ues // max(1, config.topology.sectors_per_cell)),
        },
        "scheduler": {
            "class": "sionna.sys.scheduling.PFSchedulerSUMIMO",
            "num_ut": config.ue.num_ues,
            "num_freq_res": config.nr.num_frequency_resources,
            "num_ofdm_sym": config.nr.num_ofdm_symbols,
        },
        "power_control": {
            "uplink": "open_loop_uplink_power_control",
            "downlink": "downlink_fair_power_control",
        },
        "link_adaptation": {
            "enabled": config.link_adaptation.enabled,
            "olla": config.link_adaptation.olla_enabled,
            "target_bler": config.link_adaptation.target_bler,
        },
        "phy_abstraction": {
            "class": "sionna.sys.phy_abstraction.PHYAbstraction",
            "enabled": True,
        },
    }
