from typing import Dict, Optional

import torch

from legged_gym.envs.elspider_air.elspider import ElSpider


class ElSpiderDataCollect(ElSpider):
    """ElSpider variant that exposes a rich named-field observation dict for
    offline data collection.  Policy observations (compute_observations) are
    left unchanged so a pretrained checkpoint can be used without modification.
    """

    # ------------------------------------------------------------------
    # Buffer initialisation
    # ------------------------------------------------------------------

    def _init_buffers(self):
        super()._init_buffers()
        # Build the task-identity vector once, as a [1, D] tensor on device.
        task_vec_list = getattr(getattr(self.cfg, "collect", None), "task_vec", [0.0])
        self._task_vec = torch.tensor(
            task_vec_list, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # [1, D]
        self._latest_proprio_noise = torch.zeros(
            (self.num_envs, 66), dtype=torch.float32, device=self.device
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._latest_proprio_noise[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Rich observation dict
    # ------------------------------------------------------------------

    def compute_observations(self):
        """Match ElSpider.compute_observations while caching sampled 66-dim proprio noise."""
        current_proprio = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        self.obs_buf = current_proprio.clone()
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1.0,
            ) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        self._latest_proprio_noise.zero_()
        if self.add_noise:
            obs_noise = (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec[: self.obs_buf.shape[1]]
            self.obs_buf += obs_noise
            self._latest_proprio_noise.copy_(obs_noise[:, :66])

    def _apply_obs_noise_to_diffusion_terms(
        self,
        diff_obs: Dict[str, torch.Tensor],
        obs_noise: Optional[torch.Tensor],
        scale_obs: bool,
    ) -> Dict[str, torch.Tensor]:
        """Inject the exact flat-observation noise into diffusion terms.

        ``obs_noise`` is expected to be the per-dimension residual between the
        current flat policy observation and the clean 66-dim proprio observation
        reconstructed from the same simulator state.

        If ``scale_obs`` is False, the residual is converted back to raw physical
        units before it is added to the named diffusion terms so that a later
        state-emphasis scaling step reproduces the flat obs path exactly.
        """
        if obs_noise is None:
            return diff_obs

        noise = obs_noise.to(device=self.device, dtype=torch.float32)
        if noise.ndim != 2 or noise.shape[-1] < 66:
            raise ValueError(
                f"obs_noise must have shape [N, >=66], got {tuple(noise.shape)}"
            )

        if scale_obs:
            lin_scale = ang_scale = dof_pos_scale = dof_vel_scale = 1.0
            cmd_scale = torch.ones(3, device=self.device, dtype=torch.float32)
        else:
            lin_scale = float(self.obs_scales.lin_vel)
            ang_scale = float(self.obs_scales.ang_vel)
            dof_pos_scale = float(self.obs_scales.dof_pos)
            dof_vel_scale = float(self.obs_scales.dof_vel)
            cmd_scale = self.commands_scale.to(device=self.device, dtype=torch.float32)

        diff_obs["base_lin_vel"] = diff_obs["base_lin_vel"] + noise[:, 0:3] / lin_scale
        diff_obs["base_ang_vel"] = diff_obs["base_ang_vel"] + noise[:, 3:6] / ang_scale
        diff_obs["projected_gravity"] = diff_obs["projected_gravity"] + noise[:, 6:9]
        diff_obs["commands"] = diff_obs["commands"] + noise[:, 9:12] / cmd_scale

        dof_pos_noise = noise[:, 12:30] / dof_pos_scale
        diff_obs["dof_pos"] = diff_obs["dof_pos"] + dof_pos_noise
        diff_obs["dof_pos_unbiased"] = diff_obs["dof_pos_unbiased"] + dof_pos_noise
        diff_obs["dof_vel"] = diff_obs["dof_vel"] + noise[:, 30:48] / dof_vel_scale
        return diff_obs

    def get_diffusion_observation(
        self,
        scale_obs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Return a dict of raw physical state tensors for data collection.

        All tensors live on ``self.device`` with dtype float32.
        Leading dimension is ``num_envs`` (N).

        Args:
            scale_obs: If True, apply the same obs_scales as ``compute_observations``
                       (lin_vel, ang_vel, dof_pos, dof_vel, commands scaling).
                       Useful when the downstream model expects pre-scaled inputs.
                       Defaults to False (raw physical units).
        """
        action_scale = self.cfg.control.action_scale
        # Absolute target joint position: actions are offsets from default in
        # scaled space, so the true target = actions * action_scale + default_dof_pos.
        action_unbiased = (self.actions + self.default_dof_pos/action_scale).clone()

        if scale_obs:
            base_lin_vel   = (self.base_lin_vel  * self.obs_scales.lin_vel).clone()
            base_ang_vel   = (self.base_ang_vel  * self.obs_scales.ang_vel).clone()
            projected_gravity = self.projected_gravity.clone()
            commands       = (self.commands[:, :3] * self.commands_scale).clone()
            dof_pos        = ((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos).clone()
            dof_vel        = (self.dof_vel       * self.obs_scales.dof_vel).clone()
        else:
            base_lin_vel   = self.base_lin_vel.clone()
            base_ang_vel   = self.base_ang_vel.clone()
            projected_gravity = self.projected_gravity.clone()
            commands       = self.commands[:, :3].clone()
            dof_pos        = (self.dof_pos - self.default_dof_pos).clone()
            dof_vel        = self.dof_vel.clone()

        diff_obs = {
            # [N, 3] base linear velocity in body frame, m/s
            "base_lin_vel": base_lin_vel,
            # [N, 3] base angular velocity in body frame, rad/s
            "base_ang_vel": base_ang_vel,
            # [N, 3] projected gravity vector
            "projected_gravity": projected_gravity,
            # [N, 3] commanded lin_vel_x, lin_vel_y, ang_vel_yaw
            "commands": commands,
            # [N, 18] joint position error relative to default pose
            "dof_pos": dof_pos,
            # [N, 18] absolute joint positions (raw, without default bias removed)
            "dof_pos_unbiased": self.dof_pos.clone(),
            # [N, 18] joint velocities
            "dof_vel": dof_vel,
            # [N, 18] last policy actions (raw, unscaled clipped output)
            "last_actions": self.actions.clone(),
            # [N, 6, 3] foot positions in world frame, m
            "foot_positions_world": self.foot_positions.clone(),
            # [N, 6] binary foot contact (1 = contact)
            "foot_contact": (
                self.contact_forces[:, self.feet_indices, 2] > 1.0
            ).float(),
            # [N, 6] time since last liftoff per foot, s
            "feet_air_time": self.feet_air_time.clone(),
            # [N, 18] same as action_unbiased — serves as the obs-space last_actions term
            "last_actions_unbiased": action_unbiased,
            # [N, D] fixed task-identity descriptor broadcast over all envs
            "task_vec": self._task_vec.expand(self.num_envs, -1).clone(),
        }

        diff_obs = self._apply_obs_noise_to_diffusion_terms(
            diff_obs,
            obs_noise=self._latest_proprio_noise if self.add_noise else None,
            scale_obs=scale_obs,
        )

        # Split command channels so profile weights can exactly mirror
        # commands_scale = [lin_vel, lin_vel, ang_vel].
        diff_obs["commands_xy"] = diff_obs["commands"][:, :2].clone()
        diff_obs["commands_yaw"] = diff_obs["commands"][:, 2:3].clone()
        return diff_obs
