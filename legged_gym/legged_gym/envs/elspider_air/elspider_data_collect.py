from typing import Dict

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

    # ------------------------------------------------------------------
    # Rich observation dict
    # ------------------------------------------------------------------

    def get_diffusion_observation(self, scale_obs: bool = False) -> Dict[str, torch.Tensor]:
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

        return {
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
