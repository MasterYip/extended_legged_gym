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
from legged_gym.utils.envelop.network.haa_swing_range import (
    AnalyticHaaRangeEstimator,
    HaaRangeConfig,
    HaaRangeNetwork,
    MonteCarloHaaRangeEstimator,
)


class EL_4090_ENVELOP_2(EL_4090_ENVELOP):
    """Keep the original 74-D observation while adding HAA range constraints."""

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
            if (
                network.config.condition_names != config.condition_names
                or network.config.leg_names != config.leg_names
            ):
                raise ValueError(
                    "HAA network input/output order does not match the environment: "
                    f"conditions {network.config.condition_names} != {config.condition_names}, "
                    f"legs {network.config.leg_names} != {config.leg_names}"
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
        if ranges.shape != self.haa_swing_ranges[env_ids].shape:
            raise RuntimeError(
                f"HAA estimator returned {tuple(ranges.shape)}, expected "
                f"{tuple(self.haa_swing_ranges[env_ids].shape)}"
            )
        self.haa_swing_ranges[env_ids] = ranges
        self._haa_range_condition_cache[env_ids] = condition

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        super()._resample_commands(env_ids)
        if hasattr(self, "haa_range_estimator"):
            self._refresh_haa_swing_ranges(env_ids)

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
