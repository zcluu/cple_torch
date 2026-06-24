from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from ..api import CSIShapeSpec, CSIWindow
from ..configs.schema import AdapterConfig, FeedbackConfig
from .adapters import FeedbackScheduleResult, SionnaStepResult
from .scenario import SionnaScenarioConfig


@dataclass(frozen=True)
class SionnaSysResourceState:
    resource_count_by_ue: dict[int, int]
    scheduled_mask: torch.Tensor
    achievable_rate: torch.Tensor
    topology: dict[str, object]
    channel_source: str


@dataclass(frozen=True)
class SionnaTopologySnapshot:
    ut_loc: torch.Tensor
    bs_loc: torch.Tensor
    ut_orientations: torch.Tensor
    bs_orientations: torch.Tensor
    ut_velocities: torch.Tensor
    in_state: torch.Tensor
    los: bool | None
    bs_virtual_loc: torch.Tensor
    distance_by_ue_m: torch.Tensor
    speed_by_ue_mps: torch.Tensor
    indoor_by_ue: torch.Tensor
    summary: dict[str, object]


class SionnaSysAdapter:
    """Sionna SYS-backed adapter that exposes CFR windows to CPLE flows."""

    def __init__(
        self,
        scenario: SionnaScenarioConfig,
        adapter_config: AdapterConfig,
        shape: CSIShapeSpec,
        feedback: FeedbackConfig,
        tti_ms: float,
        device: str = "cpu",
    ):
        from sionna.sys.scheduling import PFSchedulerSUMIMO

        self.scenario = scenario
        self.config = adapter_config
        self.shape = shape
        self.feedback = feedback
        self.tti_ms = tti_ms
        self.device = device
        self.generator = torch.Generator(device="cpu").manual_seed(adapter_config.seed)
        self.scheduler = PFSchedulerSUMIMO(
            num_ut=adapter_config.num_ues,
            num_freq_res=scenario.nr.num_frequency_resources,
            num_ofdm_sym=scenario.nr.num_ofdm_symbols,
            num_streams_per_ut=scenario.scheduler.num_streams_per_ue,
            beta=scenario.scheduler.beta,
            device=device,
        )
        self._channel_model = self._build_3gpp_channel_model()
        self._steps: dict[int, SionnaStepResult] = {}
        self._resource_states: dict[int, SionnaSysResourceState] = {}
        self._remaining_resources_by_slot: dict[int, dict[int, int]] = {}
        self._cfr_history: dict[int, list[torch.Tensor]] = {}
        self._rate_last_slot = torch.zeros(adapter_config.num_ues, device=device)
        self._max_generated_slot = -1
        self.reset(adapter_config.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self._steps = {}
        self._resource_states = {}
        self._remaining_resources_by_slot = {}
        self._cfr_history = {ue: [] for ue in range(self.config.num_ues)}
        self._rate_last_slot = torch.zeros(self.config.num_ues, device=self.device)
        self._max_generated_slot = -1

    def step(self, slot_idx: int) -> SionnaStepResult:
        self._ensure_step(slot_idx)
        return self._steps[slot_idx]

    def schedule_feedback(
        self,
        *,
        ue_id: int,
        request_time_ms: float,
        payload_bits: int,
    ) -> FeedbackScheduleResult:
        if payload_bits <= 0:
            raise ValueError("payload_bits must be positive")
        resource_units = math.ceil(payload_bits / self.feedback.bits_per_resource_unit)
        remaining = resource_units
        slot_idx = self._first_usable_slot(request_time_ms)
        start_time_ms: float | None = None
        finish_time_ms: float | None = None
        resource_units_used = 0

        while remaining > 0:
            self._ensure_step(slot_idx)
            available = self._available_feedback_resources(slot_idx, ue_id)
            if available > 0:
                slot_start_ms = slot_idx * self.tti_ms
                if start_time_ms is None:
                    start_time_ms = max(slot_start_ms, request_time_ms)
                used = min(remaining, available)
                self._consume_feedback_resources(slot_idx, ue_id, used)
                resource_units_used += used
                remaining -= used
                if remaining == 0:
                    finish_time_ms = slot_start_ms + (used / available) * self.tti_ms
                    finish_time_ms = max(finish_time_ms, start_time_ms)
                    break
            slot_idx += 1

        if start_time_ms is None or finish_time_ms is None:
            raise RuntimeError("feedback scheduling did not finish")
        return FeedbackScheduleResult(
            request_time_ms=request_time_ms,
            start_time_ms=start_time_ms,
            finish_time_ms=finish_time_ms,
            scheduling_delay_ms=max(0.0, start_time_ms - request_time_ms),
            feedback_duration_ms=finish_time_ms - start_time_ms,
            payload_bits=payload_bits,
            resource_units=resource_units,
            resource_units_used=resource_units_used,
            metadata={"bits_per_resource_unit": self.feedback.bits_per_resource_unit},
        )

    def _ensure_step(self, slot_idx: int) -> None:
        while self._max_generated_slot < slot_idx:
            self._generate_step(self._max_generated_slot + 1)
            self._max_generated_slot += 1

    def _generate_step(self, slot_idx: int) -> None:
        topology = self._sample_topology()
        frames_by_ue, channel_source = self._generate_cfr_frames(slot_idx, topology)
        achievable = self._achievable_from_frames(frames_by_ue, topology)
        with torch.inference_mode():
            scheduled = self.scheduler(self._rate_last_slot, achievable)
        scheduled_2d = scheduled[..., 0] if scheduled.ndim == 4 else scheduled
        resource_count = scheduled_2d.sum(dim=(0, 1)).to(torch.int64).cpu()
        resource_count_by_ue = {
            ue: int(resource_count[ue].item())
            for ue in range(self.config.num_ues)
        }
        achieved = (achievable * scheduled_2d.to(achievable.dtype)).sum(dim=(0, 1))
        normalizer = scheduled_2d.sum(dim=(0, 1)).clamp_min(1).to(achievable.dtype)
        self._rate_last_slot = achieved / normalizer

        windows = self._build_windows(frames_by_ue)
        scheduled_ues = self._select_service_ues(resource_count_by_ue)
        self._resource_states[slot_idx] = SionnaSysResourceState(
            resource_count_by_ue=resource_count_by_ue,
            scheduled_mask=scheduled.detach().cpu(),
            achievable_rate=achievable.detach().cpu(),
            topology=topology.summary,
            channel_source=channel_source,
        )
        self._steps[slot_idx] = SionnaStepResult(
            slot_idx=slot_idx,
            sim_time_ms=slot_idx * self.tti_ms,
            scheduled_ues=scheduled_ues,
            windows=windows,
            feedback_resources_by_ue=resource_count_by_ue,
            raw_state=scheduled.detach().cpu(),
            metadata={
                "adapter": "sionna_sys",
                "scheduler": "PFSchedulerSUMIMO",
                "channel_source": channel_source,
                "topology": topology.summary,
            },
        )

    def _sample_topology(self) -> SionnaTopologySnapshot:
        from sionna.sys.topology import gen_hexgrid_topology

        num_sectors = max(1, self.scenario.topology.num_cells * self.scenario.topology.sectors_per_cell)
        num_ut_per_sector = max(1, math.ceil(self.config.num_ues / num_sectors))
        kwargs = {
            "batch_size": 1,
            "num_rings": self.scenario.topology.num_rings,
            "num_ut_per_sector": num_ut_per_sector,
            "scenario": self.scenario.topology.scenario,
            "min_bs_ut_dist": self.scenario.topology.min_bs_ut_distance_m,
            "max_bs_ut_dist": self.scenario.topology.max_bs_ut_distance_m,
            "isd": self.scenario.topology.inter_site_distance_m,
            "bs_height": self.scenario.topology.bs_height_m,
            "min_ut_height": self.scenario.topology.min_ut_height_m,
            "max_ut_height": self.scenario.topology.max_ut_height_m,
            "indoor_probability": self.scenario.topology.indoor_probability,
            "min_ut_velocity": self.scenario.ue.min_velocity_mps,
            "max_ut_velocity": self.scenario.ue.max_velocity_mps,
            "device": self.device,
        }
        try:
            topology = gen_hexgrid_topology(
                **kwargs,
                los=None if self.scenario.channel.los == "auto" else bool(self.scenario.channel.los),
            )
        except TypeError:
            topology = gen_hexgrid_topology(**kwargs)

        ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state, los, bs_virtual_loc = topology[:8]
        ut_loc = ut_loc[:, : self.config.num_ues, :]
        ut_orientations = ut_orientations[:, : self.config.num_ues, :]
        ut_velocities = ut_velocities[:, : self.config.num_ues, :]
        in_state = in_state[:, : self.config.num_ues]
        bs_virtual_loc = bs_virtual_loc[:, :, : self.config.num_ues, :]
        distances = self._bs_ut_distance_by_ue(ut_loc, bs_loc)
        speeds = torch.linalg.norm(ut_velocities[0], dim=-1)
        indoor = in_state[0].to(torch.bool)
        summary = {
            "num_bs": int(bs_loc.shape[1]),
            "num_ues": self.config.num_ues,
            "mean_bs_ut_distance_m": float(distances.mean().item()),
            "mean_ut_speed_mps": float(speeds.mean().item()),
            "indoor_ratio": float(indoor.to(torch.float32).mean().item()),
            "los_mode": "auto" if los is None else "fixed",
        }
        return SionnaTopologySnapshot(
            ut_loc=ut_loc.detach().cpu(),
            bs_loc=bs_loc.detach().cpu(),
            ut_orientations=ut_orientations.detach().cpu(),
            bs_orientations=bs_orientations.detach().cpu(),
            ut_velocities=ut_velocities.detach().cpu(),
            in_state=in_state.detach().cpu(),
            los=los,
            bs_virtual_loc=bs_virtual_loc.detach().cpu(),
            distance_by_ue_m=distances.detach().cpu(),
            speed_by_ue_mps=speeds.detach().cpu(),
            indoor_by_ue=indoor.detach().cpu(),
            summary=summary,
        )

    def _generate_cfr_frames(
        self,
        slot_idx: int,
        topology: SionnaTopologySnapshot,
    ) -> tuple[dict[int, torch.Tensor], str]:
        if self._channel_model is None:
            if self.scenario.channel.use_3gpp_channel:
                raise ValueError(
                    f"TR 38.901 channel model is not available for {self.scenario.channel.model}"
                )
            return self._fallback_cfr_frames(slot_idx, topology), "topology_aware_fallback"

        from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies

        self._channel_model.set_topology(
            topology.ut_loc.to(self.device),
            topology.bs_loc.to(self.device),
            topology.ut_orientations.to(self.device),
            topology.bs_orientations.to(self.device),
            topology.ut_velocities.to(self.device),
            topology.in_state.to(self.device),
            topology.los,
            topology.bs_virtual_loc.to(self.device),
        )
        coefficients, delays = self._channel_model(
            num_time_samples=self.shape.output_frames,
            sampling_frequency=self._sampling_frequency_hz(),
        )
        frequencies = subcarrier_frequencies(
            self.scenario.nr.num_frequency_resources,
            self.scenario.nr.subcarrier_spacing_hz,
            device=self.device,
        )
        cfr = cir_to_ofdm_channel(
            frequencies,
            coefficients,
            delays,
            normalize=self.scenario.channel.normalize_channel,
        )
        frames = {}
        for ue in range(self.config.num_ues):
            ue_cfr = cfr[0, :, :, ue, :, :, :]
            frame_series = self._map_cfr_to_user_frames(ue_cfr)
            frames[ue] = frame_series.detach().cpu()
        return frames, "3gpp_tr38901_cfr"

    def _fallback_cfr_frames(
        self,
        slot_idx: int,
        topology: SionnaTopologySnapshot,
    ) -> dict[int, torch.Tensor]:
        frames: dict[int, torch.Tensor] = {}
        for ue in range(self.config.num_ues):
            base = torch.randn(
                (self.shape.output_frames, *self.shape.frame_shape),
                generator=self.generator,
                dtype=torch.float32,
            )
            distance = topology.distance_by_ue_m[ue].clamp_min(1.0)
            speed = topology.speed_by_ue_mps[ue]
            gain = (distance / topology.distance_by_ue_m.min().clamp_min(1.0)).pow(-0.35)
            phase = 0.01 * slot_idx + 0.001 * speed
            frames[ue] = gain * (base + phase)
        return frames

    def _map_cfr_to_user_frames(self, ue_cfr: torch.Tensor) -> torch.Tensor:
        # ue_cfr: [num_bs, num_rx_ant, num_tx_ant, time, frequency]
        time_axis = ue_cfr.shape[-2]
        if self.shape.dtype == "float32" and torch.is_complex(ue_cfr):
            source = torch.view_as_real(ue_cfr)
        elif self.shape.dtype == "complex64" and not torch.is_complex(ue_cfr):
            source = torch.complex(ue_cfr, torch.zeros_like(ue_cfr))
        else:
            source = ue_cfr
        flattened = source.permute(3, 0, 1, 2, 4, *range(5, source.ndim)).reshape(time_axis, -1)
        needed = self.shape.elements_per_frame
        if flattened.shape[1] < needed:
            repeats = math.ceil(needed / flattened.shape[1])
            flattened = flattened.repeat(1, repeats)
        flattened = flattened[:, :needed]
        return flattened.reshape(self.shape.output_frames, *self.shape.frame_shape)

    def _achievable_from_frames(
        self,
        frames_by_ue: dict[int, torch.Tensor],
        topology: SionnaTopologySnapshot,
    ) -> torch.Tensor:
        shape = (
            self.scenario.nr.num_ofdm_symbols,
            self.scenario.nr.num_frequency_resources,
            self.config.num_ues,
        )
        achievable = torch.empty(shape, device=self.device)
        for ue in range(self.config.num_ues):
            frame = frames_by_ue[ue]
            power = frame.abs().float().pow(2).mean().clamp_min(1e-6)
            speed = topology.speed_by_ue_mps[ue].to(self.device)
            mobility_penalty = 1.0 / (1.0 + 0.03 * speed)
            small_scale = torch.rand(shape[:2], generator=self.generator, device="cpu").to(self.device)
            achievable[:, :, ue] = power.to(self.device) * mobility_penalty * (0.5 + small_scale)
        return achievable

    def _build_windows(self, frames_by_ue: dict[int, torch.Tensor]) -> dict[int, CSIWindow]:
        windows: dict[int, CSIWindow] = {}
        for ue, frame_series in frames_by_ue.items():
            current = frame_series[0]
            self._cfr_history[ue].append(current)
            self._cfr_history[ue] = self._cfr_history[ue][-self.shape.history_len :]
            history = list(self._cfr_history[ue])
            while len(history) < self.shape.history_len:
                history.insert(0, history[0])
            window = CSIWindow(
                history=torch.stack(history, dim=0),
                current=current,
                target=frame_series,
                raw={"source": "cfr"},
            )
            window.validate(self.shape)
            windows[ue] = window
        return windows

    def _available_feedback_resources(self, slot_idx: int, ue_id: int) -> int:
        if slot_idx not in self._remaining_resources_by_slot:
            self._ensure_step(slot_idx)
            self._remaining_resources_by_slot[slot_idx] = dict(
                self._resource_states[slot_idx].resource_count_by_ue
            )
        return self._remaining_resources_by_slot[slot_idx].get(ue_id, 0)

    def _consume_feedback_resources(self, slot_idx: int, ue_id: int, units: int) -> None:
        if slot_idx not in self._remaining_resources_by_slot:
            self._remaining_resources_by_slot[slot_idx] = dict(
                self._resource_states[slot_idx].resource_count_by_ue
            )
        self._remaining_resources_by_slot[slot_idx][ue_id] = max(
            0,
            self._remaining_resources_by_slot[slot_idx].get(ue_id, 0) - units,
        )

    def _select_service_ues(self, resource_count_by_ue: dict[int, int]) -> list[int]:
        ranked = sorted(
            resource_count_by_ue,
            key=lambda ue: (-resource_count_by_ue[ue], ue),
        )
        selected = [ue for ue in ranked if resource_count_by_ue[ue] > 0]
        return selected[: self.config.scheduled_per_slot]

    def _bs_ut_distance_by_ue(self, ut_loc: torch.Tensor, bs_loc: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(ut_loc[0], bs_loc[0])
        return distances.min(dim=1).values

    def _build_3gpp_channel_model(self):
        if not self.scenario.channel.use_3gpp_channel:
            return None
        if self.scenario.channel.model not in {"umi", "uma", "rma"}:
            return None
        from sionna.phy.channel.tr38901 import PanelArray, RMa, UMa, UMi

        carrier = self.scenario.nr.carrier_frequency_hz
        ut_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier,
            device=self.device,
        )
        bs_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=carrier,
            device=self.device,
        )
        direction = "uplink" if self.scenario.nr.duplex_mode == "uplink" else "downlink"
        common = {
            "carrier_frequency": carrier,
            "ut_array": ut_array,
            "bs_array": bs_array,
            "direction": direction,
            "enable_pathloss": self.scenario.channel.enable_pathloss,
            "enable_shadow_fading": self.scenario.channel.enable_shadow_fading,
            "device": self.device,
        }
        if self.scenario.channel.model == "umi":
            return UMi(o2i_model=self.scenario.channel.o2i_model, **common)
        if self.scenario.channel.model == "uma":
            return UMa(o2i_model=self.scenario.channel.o2i_model, **common)
        return RMa(**common)

    def _sampling_frequency_hz(self) -> float:
        return max(1.0, self.scenario.nr.subcarrier_spacing_hz * self.scenario.nr.num_ofdm_symbols)

    def _first_usable_slot(self, request_time_ms: float) -> int:
        slot_float = request_time_ms / self.tti_ms
        slot_idx = int(math.floor(slot_float))
        if request_time_ms > slot_idx * self.tti_ms + 1e-9:
            return slot_idx + 1
        return slot_idx
