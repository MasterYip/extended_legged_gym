"""Envelope-conditioned EL4090 with generated per-leg HAA swing ranges."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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
    """66-D locomotion policy plus a separate envelope-to-HAA network path."""

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
        range_cfg = self.cfg.haa_swing_range
        haa_leg_names = tuple(self.dof_names[index].split("_", 1)[0] for index in self.haa_indices.tolist())
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

    def _get_noise_scale_vec(self, cfg) -> torch.Tensor:
        """Noise vector for the 66-D observation without envelope condition."""
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
        return noise_vec

    def compute_observations(self) -> None:
        """Policy observes only proprioception and [vx, vy, yaw_rate]."""
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

    def _get_haa_moving_mask(self) -> torch.Tensor:
        threshold = float(getattr(self.cfg.rewards, "haa_range_min_command", 0.15))
        angular_index = 3 if self.cfg.commands.heading_command else 2
        return torch.logical_or(
            torch.norm(self.commands[:, :2], dim=1) > threshold,
            torch.abs(self.commands[:, angular_index]) > threshold,
        ).float()

    def _reward_haa_swing_in_range(self) -> torch.Tensor:
        """Reward useful HAA excursion and velocity only while inside its range."""
        lower = self.haa_swing_ranges[..., 0]
        upper = self.haa_swing_ranges[..., 1]
        position = self.dof_pos[:, self.haa_indices]
        center = 0.5 * (lower + upper)
        half_range = (0.5 * (upper - lower)).clamp_min(1e-6)
        inside = torch.logical_and(position >= lower, position <= upper).float()
        excursion = torch.clamp(torch.abs(position - center) / half_range, 0.0, 1.0)
        velocity_clip = max(float(getattr(self.cfg.rewards, "haa_range_velocity_clip", 5.0)), 1e-6)
        velocity = torch.clamp(torch.abs(self.dof_vel[:, self.haa_indices]) / velocity_clip, 0.0, 1.0)
        swing = torch.mean(inside * excursion * velocity, dim=1)
        return swing * self._get_haa_moving_mask() * self._get_gait_reward_active_mask()

    def _reward_haa_range_violation(self) -> torch.Tensor:
        """Quadratic hard-bound penalty for each HAA outside its generated range."""
        margin = float(getattr(self.cfg.rewards, "haa_range_margin", 0.0))
        lower = self.haa_swing_ranges[..., 0] - margin
        upper = self.haa_swing_ranges[..., 1] + margin
        position = self.dof_pos[:, self.haa_indices]
        violation = torch.relu(lower - position) + torch.relu(position - upper)
        return torch.mean(torch.square(violation), dim=1)
