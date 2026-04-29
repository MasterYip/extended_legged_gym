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

    def get_diffusion_observation(self) -> Dict[str, torch.Tensor]:
        """Return a dict of raw physical state tensors for data collection.

        All tensors live on ``self.device`` with dtype float32.
        Leading dimension is ``num_envs`` (N).
        """
        return {
            # [N, 3] base linear velocity in body frame, m/s
            "base_lin_vel": self.base_lin_vel.clone(),
            # [N, 3] base angular velocity in body frame, rad/s
            "base_ang_vel": self.base_ang_vel.clone(),
            # [N, 3] projected gravity vector
            "projected_gravity": self.projected_gravity.clone(),
            # [N, 3] commanded lin_vel_x, lin_vel_y, ang_vel_yaw (raw, unscaled)
            "commands": self.commands[:, :3].clone(),
            # [N, 18] joint position error relative to default pose, rad
            "dof_pos": (self.dof_pos - self.default_dof_pos).clone(),
            # [N, 18] joint velocities, rad/s
            "dof_vel": self.dof_vel.clone(),
            # [N, 18] last policy actions
            "last_actions": self.actions.clone(),
            # [N, 6, 3] foot positions in world frame, m
            "foot_positions_world": self.foot_positions.clone(),
            # [N, 6] binary foot contact (1 = contact)
            "foot_contact": (
                self.contact_forces[:, self.feet_indices, 2] > 1.0
            ).float(),
            # [N, 6] time since last liftoff per foot, s
            "feet_air_time": self.feet_air_time.clone(),
            # [N, D] fixed task-identity descriptor broadcast over all envs
            "task_vec": self._task_vec.expand(self.num_envs, -1).clone(),
        }
