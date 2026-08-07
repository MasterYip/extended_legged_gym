"""Envelope-conditioned EL4090 with generated per-leg HAA swing ranges."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import torch
from isaacgym import gymtorch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.el_4090.spider_envelop.el_4090 import EL_4090_ENVELOP
from legged_gym.envs.el_4090.spider_envelop_2.el4090_spider_config import (
    El4090Envelop2Cfg,
)
from legged_gym.envs.el_4090.spider_envelop_2.envelope_condition import (
    EnvelopeConditionState,
)
from legged_gym.utils.envelop.network.haa_swing_range import (
    AnalyticHaaRangeEstimator,
    HaaRangeConfig,
    HaaRangeNetwork,
    MonteCarloHaaRangeEstimator,
)


class EL_4090_ENVELOP_2(EL_4090_ENVELOP):
    """83-D locomotion policy with explicit HAA ranges and gait phase."""

    cfg: El4090Envelop2Cfg

    def __init__(
        self,
        cfg: El4090Envelop2Cfg,
        sim_params,
        physics_engine,
        sim_device,
        headless,
        task_name: str = "el4090_envelop_2",
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, task_name)

    def _init_buffers(self) -> None:
        super()._init_buffers()
        # Keep one action older than the inherited ``last_actions`` buffer so
        # the reward can penalize the discrete second action difference.
        self.last_last_actions = torch.zeros_like(self.actions)
        self.action_history_steps = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.envelope_state = EnvelopeConditionState(
            self.cfg.envelope,
            self.num_envs,
            self.device,
        )
        # Preserve the names used by inherited morphology/reward helpers, but
        # source their values from envelope_state rather than commands.
        self.condition_names = list(self.envelope_state.condition_names)
        self.condition_dim = self.envelope_state.condition_dim
        self.condition_low = self.envelope_state.low
        self.condition_high = self.envelope_state.high
        self.morphology_prior_obs_indices = torch.tensor(
            [
                self.condition_names.index("morphology_front_prior"),
                self.condition_names.index("morphology_middle_prior"),
                self.condition_names.index("morphology_back_prior"),
            ],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        range_cfg = self.cfg.haa_swing_range
        haa_leg_names = tuple(self.dof_names[index].split("_", 1)[0] for index in self.haa_indices.tolist())
        expected_legs = {"LB", "LF", "LM", "RB", "RF", "RM"}
        if set(haa_leg_names) != expected_legs:
            raise ValueError(f"Unexpected HAA leg order: {haa_leg_names}")
        second_tripod = {"LM", "RB", "RF"}
        self.haa_phase_offsets = torch.tensor(
            [math.pi if leg_name in second_tripod else 0.0 for leg_name in haa_leg_names],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        spider_haa = tuple(float(self.default_dof_pos[0, index]) for index in self.haa_indices.tolist())
        mammal_haa = tuple(float(self.mammal_default_dof_pos[0, index]) for index in self.haa_indices.tolist())
        estimator_cfg = HaaRangeConfig(
            condition_names=tuple(self.condition_names),
            leg_names=haa_leg_names,
            joint_lower=float(range_cfg.joint_lower),
            joint_upper=float(range_cfg.joint_upper),
            leg_reach=float(range_cfg.leg_reach),
            front_hip_offset=float(range_cfg.front_hip_offset),
            middle_hip_offset=float(range_cfg.middle_hip_offset),
            back_hip_offset=float(range_cfg.back_hip_offset),
            spider_swing_limit=float(range_cfg.spider_swing_limit),
            mammal_swing_limit=float(range_cfg.mammal_swing_limit),
            minimum_half_range=float(range_cfg.minimum_half_range),
            spider_haa=spider_haa,
            mammal_haa=mammal_haa,
        )
        self._haa_range_output_indices = torch.arange(
            len(haa_leg_names),
            dtype=torch.long,
            device=self.device,
        )
        self.haa_range_estimator = self._make_haa_range_estimator(estimator_cfg)
        self.haa_swing_ranges = torch.zeros(
            self.num_envs,
            len(haa_leg_names),
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self._haa_range_condition_cache = torch.full(
            (self.num_envs, self.condition_dim),
            float("nan"),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self._refresh_haa_swing_ranges()

    def step(self, actions):
        """Step the simulator while preserving two valid action-history frames."""
        previous_actions = self.last_actions.clone()
        result = super().step(actions)
        self._update_action_history(previous_actions, result[3])
        return result

    @torch.no_grad()
    def _update_action_history(
        self,
        previous_actions: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Advance action history and discard history across episode resets."""
        done_mask = dones.to(dtype=torch.bool)
        active_mask = ~done_mask
        self.last_last_actions[:] = previous_actions
        self.action_history_steps[active_mask] = torch.clamp(
            self.action_history_steps[active_mask] + 1,
            max=2,
        )
        self.last_last_actions[done_mask] = 0.0
        self.last_actions[done_mask] = 0.0
        self.action_history_steps[done_mask] = 0

    def _get_noise_scale_vec(self, cfg) -> torch.Tensor:
        """Noise vector for the 83-D envelop_2 policy observation."""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.0
        noise_vec[12:30] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[30:48] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[48:66] = 0.0
        noise_vec[66:69] = 0.0
        noise_vec[69:81] = 0.0
        noise_vec[81:83] = 0.0
        return noise_vec

    def compute_observations(self) -> None:
        """Policy observes proprioception, morphology, HAA ranges, and gait phase."""
        haa_center, haa_half_range = self._get_haa_range_center_and_half()
        gait_phase = self._get_gait_phase()
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, 0:1] * self.command_lin_vel_scale,
                self.commands[:, 1:2] * self.command_lin_vel_scale,
                self.commands[:, 2:3] * self.command_ang_vel_scale,
                (self.dof_pos - self.embedded_state_default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                self._get_structure_condition().index_select(1, self.morphology_prior_obs_indices)
                * self.obs_scales.morphology_prior,
                haa_center * self.obs_scales.haa_range_center,
                haa_half_range * self.obs_scales.haa_range_half,
                torch.stack((torch.sin(gait_phase), torch.cos(gait_phase)), dim=1)
                * self.obs_scales.gait_phase,
            ),
            dim=-1,
        )
        if getattr(self.cfg.lidar, "enable", False):
            self.obs_buf = torch.cat((self.obs_buf, self._get_lidar_observations()), dim=-1)
        elif self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.45 - self.measured_heights,
                -5,
                5.0,
            ) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_structure_condition(self) -> torch.Tensor:
        """Return the external envelope state; never read it from commands."""
        return self.envelope_state.get()

    def _get_haa_range_center_and_half(self) -> Tuple[torch.Tensor, torch.Tensor]:
        lower = self.haa_swing_ranges[..., 0]
        upper = self.haa_swing_ranges[..., 1]
        return 0.5 * (lower + upper), 0.5 * (upper - lower)

    def _get_gait_phase(self) -> torch.Tensor:
        period = max(float(getattr(self.cfg.rewards, "haa_phase_period", 0.5)), 1e-6)
        episode_time = self.episode_length_buf.to(dtype=torch.float) * self.dt
        return torch.remainder(2.0 * math.pi * episode_time / period, 2.0 * math.pi)

    def _get_haa_movement_strength(self) -> torch.Tensor:
        min_command = float(getattr(self.cfg.rewards, "haa_phase_min_command", 0.1))
        full_command = float(getattr(self.cfg.rewards, "haa_phase_full_command", 0.5))
        yaw_weight = float(getattr(self.cfg.rewards, "haa_phase_yaw_weight", 0.5))
        command_magnitude = (
            torch.norm(self.commands[:, :2], dim=1)
            + yaw_weight * torch.abs(self.commands[:, 2])
        )
        return torch.clamp(
            (command_magnitude - min_command) / max(full_command - min_command, 1e-6),
            0.0,
            1.0,
        )

    def _get_haa_phase_target(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return target, availability, and command strength for each environment."""
        center, half_range = self._get_haa_range_center_and_half()
        min_half_range = float(self.cfg.haa_swing_range.minimum_half_range)
        full_half_range = float(self.cfg.haa_swing_range.spider_swing_limit)
        availability = torch.clamp(
            (half_range - min_half_range) / max(full_half_range - min_half_range, 1e-6),
            0.0,
            1.0,
        )
        movement_strength = self._get_haa_movement_strength()
        phase = self._get_gait_phase().unsqueeze(1) + self.haa_phase_offsets.unsqueeze(0)
        amplitude_ratio = torch.clamp(
            half_range.new_tensor(float(getattr(self.cfg.rewards, "haa_phase_amplitude_ratio", 0.6))),
            0.0,
            1.0,
        )
        amplitude = (
            movement_strength.unsqueeze(1)
            * availability
            * amplitude_ratio
            * half_range
        )
        target = center + amplitude * torch.sin(phase)
        return target, availability, movement_strength

    def _make_haa_range_estimator(self, config: HaaRangeConfig):
        range_cfg = self.cfg.haa_swing_range
        method = str(range_cfg.method).lower()
        if method == "analytic":
            return AnalyticHaaRangeEstimator(config)
        if method == "monte_carlo":
            return MonteCarloHaaRangeEstimator(
                config,
                num_samples=int(range_cfg.monte_carlo_samples),
                quantile=float(range_cfg.monte_carlo_quantile),
                seed=range_cfg.monte_carlo_seed,
            )
        if method == "network":
            checkpoint = str(range_cfg.network_checkpoint)
            if not checkpoint:
                raise ValueError("haa_swing_range.network_checkpoint is required for method='network'")
            checkpoint = checkpoint.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            network = HaaRangeNetwork.from_checkpoint(Path(checkpoint), device=self.device)
            if network.config.condition_names != config.condition_names:
                raise ValueError(
                    "HAA network input order does not match the environment: "
                    f"{network.config.condition_names} != {config.condition_names}"
                )
            missing_legs = set(config.leg_names).difference(network.config.leg_names)
            if missing_legs:
                raise ValueError(f"HAA network is missing legs: {sorted(missing_legs)}")
            # Keep the checkpoint output contract exactly as trained, then map
            # RF/RM/RB/LF/LM/LB to Isaac Gym's internal HAA DOF order.
            self._haa_range_output_indices = torch.tensor(
                [network.config.leg_names.index(name) for name in config.leg_names],
                dtype=torch.long,
                device=self.device,
            )
            return network
        raise ValueError(
            f"Unknown haa_swing_range.method={range_cfg.method!r}; "
            "expected 'analytic', 'monte_carlo', or 'network'"
        )

    @torch.no_grad()
    def _refresh_haa_swing_ranges(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            condition = self._get_structure_condition()
            changed = torch.any(condition != self._haa_range_condition_cache, dim=1)
            env_ids = changed.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        condition = self._get_structure_condition()[env_ids]
        ranges = self.haa_range_estimator(condition)
        ranges = ranges[:, self._haa_range_output_indices]
        if ranges.shape != self.haa_swing_ranges[env_ids].shape:
            raise RuntimeError(
                f"HAA estimator returned {tuple(ranges.shape)}, expected "
                f"{tuple(self.haa_swing_ranges[env_ids].shape)}"
            )
        self.haa_swing_ranges[env_ids] = ranges
        self._haa_range_condition_cache[env_ids] = condition

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        for command_index, range_name in enumerate(
            ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")
        ):
            low, high = self.command_ranges[range_name]
            self.commands[env_ids, command_index] = low + torch.rand(
                env_ids.numel(),
                dtype=torch.float,
                device=self.device,
            ) * (high - low)
        self.envelope_state.sample(env_ids)
        if getattr(self.cfg.rewards, "reset_structure_transition_on_resample", True):
            self.embedded_state_transition_time[env_ids] = 0.0
            self.filtered_embedded_state_default_dof_pos[env_ids] = self.default_dof_pos
        if hasattr(self, "haa_range_estimator"):
            self._refresh_haa_swing_ranges(env_ids)

    @torch.no_grad()
    def set_envelope_condition(
        self,
        values,
        env_ids: Optional[torch.Tensor] = None,
        *,
        derive_priors: bool = True,
    ) -> torch.Tensor:
        """Public hook for a future external envelope computation module."""
        updated = self.envelope_state.set(
            values,
            env_ids,
            derive_priors=derive_priors,
        )
        target_ids = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None
            else env_ids.to(device=self.device, dtype=torch.long)
        )
        self._refresh_haa_swing_ranges(target_ids)
        self.embedded_state_default_dof_pos[target_ids] = self._get_condition_target_dof_pos()[
            target_ids
        ]
        return updated

    def _post_physics_step_callback(self) -> None:
        super()._post_physics_step_callback()
        self._refresh_haa_swing_ranges()

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """P control around the current morphology preset (without low-pass)."""
        if self.cfg.control.control_type != "P":
            return super()._compute_torques(actions)
        actions_scaled = actions * self.cfg.control.action_scale
        target = actions_scaled + self.embedded_state_default_dof_pos
        torques = self.p_gains * (target - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset directly at the newly sampled P-controller morphology preset."""
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.last_last_actions[env_ids] = 0.0
        self.action_history_steps[env_ids] = 0
        target = self._get_condition_target_dof_pos()[env_ids]
        self.embedded_state_default_dof_pos[env_ids] = target
        self.dof_pos[env_ids] = target
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reward_haa_range_violation(self) -> torch.Tensor:
        """Penalize each HAA quadratically when it leaves its generated range."""
        margin = float(getattr(self.cfg.rewards, "haa_range_margin", 0.0))
        lower = self.haa_swing_ranges[..., 0] - margin
        upper = self.haa_swing_ranges[..., 1] + margin
        position = self.dof_pos[:, self.haa_indices]
        violation = torch.relu(lower - position) + torch.relu(position - upper)
        half_range = (0.5 * (upper - lower)).clamp_min(1e-6)
        normalized_violation = violation / half_range
        return torch.mean(torch.square(normalized_violation), dim=1)

    def _reward_action_jerk(self) -> torch.Tensor:
        """Penalize rapid action reversals using the discrete second difference."""
        second_difference = (
            self.actions - 2.0 * self.last_actions + self.last_last_actions
        )
        valid_history = (self.action_history_steps >= 2).to(dtype=self.actions.dtype)
        return torch.sum(torch.square(second_difference), dim=1) * valid_history

    def _reward_haa_phase_tracking(self) -> torch.Tensor:
        """Track a smooth, range-aware tripod HAA target without rewarding velocity."""
        target, _, movement_strength = self._get_haa_phase_target()
        _, half_range = self._get_haa_range_center_and_half()
        position = self.dof_pos[:, self.haa_indices]
        normalized_error = (position - target) / half_range.clamp_min(1e-6)
        return torch.mean(torch.square(normalized_error), dim=1) * movement_strength
