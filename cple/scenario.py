from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .config import MockAdapterConfig, PlatformConfig


ScenarioName = Literal["umi", "uma", "rma", "inh"]
DuplexMode = Literal["uplink", "downlink"]
SchedulerType = Literal["proportional_fair", "round_robin", "fixed"]


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
    csi_dim: int = 64
    history_len: int = 4
    deadline_ms: float = 5.0
    feedback_capacity_per_slot: int = 4
    feedback_slot_ms: float | None = None


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
    if config.cple.csi_dim <= 0:
        raise ValueError("cple.csi_dim must be positive")
    if config.cple.history_len <= 0:
        raise ValueError("cple.history_len must be positive")
    if config.cple.feedback_capacity_per_slot <= 0:
        raise ValueError("cple.feedback_capacity_per_slot must be positive")
    if config.cple.feedback_slot_ms is not None and config.cple.feedback_slot_ms <= 0:
        raise ValueError("cple.feedback_slot_ms must be positive when set")
    if config.nr.carrier_frequency_hz <= 0:
        raise ValueError("nr.carrier_frequency_hz must be positive")
    if config.nr.bandwidth_hz <= 0:
        raise ValueError("nr.bandwidth_hz must be positive")
    if config.scheduler.type not in {"proportional_fair", "round_robin", "fixed"}:
        raise ValueError("scheduler.type must be proportional_fair, round_robin, or fixed")


def scenario_to_mock_adapter_config(config: SionnaScenarioConfig) -> MockAdapterConfig:
    return MockAdapterConfig(
        num_ues=config.ue.num_ues,
        csi_dim=config.cple.csi_dim,
        scheduled_per_slot=config.ue.scheduled_per_slot,
        history_len=config.cple.history_len,
        seed=config.seed,
    )


def scenario_to_platform_config(config: SionnaScenarioConfig, output_dir: str | None = None) -> PlatformConfig:
    return PlatformConfig(
        run_id=config.name,
        num_slots=config.time.num_slots,
        tti_ms=config.time.tti_ms,
        deadline_ms=config.cple.deadline_ms,
        device="cpu",
        feedback_capacity_per_slot=config.cple.feedback_capacity_per_slot,
        feedback_slot_ms=config.cple.feedback_slot_ms,
        warmup_slots=config.time.warmup_slots,
        output_dir=output_dir or f"outputs/{config.name}",
        seed=config.seed,
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
            "class": "sionna.sys.scheduling.PFSchedulerSUMIMO"
            if config.scheduler.type == "proportional_fair"
            else config.scheduler.type,
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
