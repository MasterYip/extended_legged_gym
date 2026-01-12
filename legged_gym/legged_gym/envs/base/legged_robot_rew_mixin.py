from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from legged_gym.utils.helpers import class_to_dict

class LeggedRobotRewMixin:
    """
    Mixin class for batch rollout with reward calculation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speed_min = 0.1  # Minimum speed for some rewards

    def _get_reward_scales(self, stage=0):
        self.reward_scales_dict = class_to_dict(self.cfg.rewards.scales)
        if self.cfg.rewards.multi_stage_rewards:
            reward_scales = {}
            for key, value in self.reward_scales_dict.items():
                if not isinstance(value, list):
                    reward_scales[key] = value
                else:
                    if stage >= len(value):
                        reward_scales[key] = value[-1]
                    else:
                        reward_scales[key] = value[stage]
            return reward_scales
        else:
            return self.reward_scales_dict

    def update_reward_scales(self, mean_reward):
        if mean_reward > self.cfg.rewards.reward_stage_threshold and \
                self.reward_scales_stage < self.cfg.rewards.reward_max_stage:
            self.reward_scales_stage += 1
            self.reward_scales = self._get_reward_scales(self.reward_scales_stage)
            self._prepare_reward_function()
            return True
        return False

    # ------------ base penalty ------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_base_foot_height(self):
        # Penalize base height away from target using contact feet as ground reference
        # Get contact status for feet (z-force > threshold indicates contact)
        contact = self.feet_contact_time > 1e-3

        foot_heights = self.foot_positions[:, :, 2]  # Z positions of feet

        # Calculate mean height of contact feet for each environment
        contact_foot_heights = torch.where(contact, foot_heights, torch.nan)
        
        # Use nanmean to compute mean only for contact feet, fallback to base height if no contacts
        estimated_ground_height = torch.nanmean(contact_foot_heights, dim=1)
        
        # Handle case where no feet are in contact (all NaN) - use current base height as reference
        no_contact_mask = torch.isnan(estimated_ground_height)
        estimated_ground_height = torch.where(no_contact_mask, 
                                            self.root_states[:, 2] - self.cfg.rewards.base_height_target,
                                            estimated_ground_height)
        
        # Calculate base height relative to estimated ground
        base_height_relative = self.root_states[:, 2] - estimated_ground_height
        
        # Penalize deviation from target base height
        return torch.square(base_height_relative - self.cfg.rewards.base_height_target)

    # ------------ joint penalty ------------
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # ------------ joint limit penalty ------------
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    # ------------ collision penalty ------------
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_feet_stumble_liftup(self):
        # Reward feet stumbling and lifting up
        feet_stumble = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2])
        feet_zvel = self.foot_velocities[:, :, 2]
        return torch.sum(feet_stumble * feet_zvel, dim=1)

    # ------------ contact penalty ------------
    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # TODO: Last contact should not be updated by _reward_feet_air_time
        contact_filt = torch.logical_or(contact, self.last_contacts)
        # xy lin vel
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip

    def _reward_jump_air(self):
        # Penalize less than 3 feet on the ground
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # TODO: Last contact should not be updated by _reward_feet_air_time
        contact_filt = torch.logical_or(contact, self.last_contacts)
        return torch.clip(torch.sum((~contact_filt) * (self.feet_air_time - 0.5), dim=1) - len(self.feet_indices)/2, 0.)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        self.feet_contact_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        self.feet_contact_time *= contact_filt
        return rew_airTime

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    # ------------ 2 step gait penalty ------------
    def _reward_gait_2_step(self):
        # Foot index: FL, FR, RL, RR
        sync_fl_rr = self._sync_reward_func(0, 3)
        sync_fr_rl = self._sync_reward_func(1, 2)
        sync_reward = (sync_fl_rr + sync_fr_rl) / 2
        async_fl_fr = self._async_reward_func(0, 1)
        async_fl_rl = self._async_reward_func(0, 2)
        async_rr_rl = self._async_reward_func(3, 2)
        async_rr_fr = self._async_reward_func(3, 1)
        async_reward = (async_fl_fr + async_fl_rl + async_rr_rl + async_rr_fr) / 4
        re = sync_reward + async_reward
        if self.cfg.commands.heading_command:
            re = re * torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > self.speed_min, 
                                    torch.abs(self.commands[:, 3]) >= self.speed_min/ 2)
        else:
            re = re * torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > self.speed_min, 
                                    torch.abs(self.commands[:, 2]) >= self.speed_min/ 2)
        return re
    
    def _sync_reward_func(self, foot_0: int, foot_1: int, max_err=2) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.feet_air_time
        contact_time = self.feet_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=max_err**2)
        return se_air + se_contact
    
    def _async_reward_func(self, foot_0: int, foot_1: int, max_err=2) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.feet_air_time
        contact_time = self.feet_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=max_err**2)
        return se_act_0 + se_act_1
    
    def _reward_four_footup(self):
        foot_up = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) * 0.1
        self.all_feet_up = torch.all(self.contact_forces[:, self.feet_indices, 2] < 1, dim=1)
        foot_up = foot_up * self.all_feet_up
        return foot_up

    # ------------ misc penalty ------------
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < self.speed_min)


    # ------------ base reward ------------

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes in base frame)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)



