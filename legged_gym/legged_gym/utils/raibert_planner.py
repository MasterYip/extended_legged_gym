'''
Author: Raymon Yip 2205929492@qq.com
Date: 2025-03-14 19:57:09
Description: file content
FilePath: /legged_gym_cmp/legged_gym/legged_gym/utils/raibert_planner.py
LastEditTime: 2025-03-18 17:01:45
LastEditors: Raymon Yip
'''

import torch
from isaacgym.torch_utils import quat_apply, quat_rotate, quat_rotate_inverse, \
    quat_from_angle_axis, to_torch, quat_mul, quat_conjugate, normalize
import numpy as np
from .math_utils import RandomWalker, cubic_hermite_evaluate, ypr_to_quat


def sin_swing_traj(swing_height, phase: torch.Tensor):
    # if 0 < phase < 0.5: sin swing trajectory, else 0
    return torch.where(phase < 0.5, swing_height * torch.sin(2 * torch.pi * phase), torch.zeros_like(phase))


class SimpleRaibertPlannerConfig(object):

    dt = 0.02

    # Nominal Base and Foot
    nominal_y_shift = 0.06
    nominal_x_shift = 0.0
    # Origin Leg order: RF, RM, RB, LF, LM, LB
    nominal_foothold_base_origin_index = [
        [0.354 + nominal_x_shift, -0.28 - nominal_y_shift, -0.28],
        [0.054 + nominal_x_shift, -0.34 - nominal_y_shift, -0.28],
        [-0.354 + nominal_x_shift, -0.28 - nominal_y_shift, -0.28],
        [0.354 + nominal_x_shift, 0.28 + nominal_y_shift, -0.28],
        [0.054 + nominal_x_shift, 0.34 + nominal_y_shift, -0.28],
        [-0.354 + nominal_x_shift, 0.28 + nominal_y_shift, -0.28],
    ]
    # URDF alphabet order: LB, LF, LM, RB, RF, RM
    foothold_index_remap = [5, 3, 4, 2, 0, 1]
    nominal_base_height = 0.30

    # Gait
    gait_period = 0.5
    foot_phases_origin = [0.0, 0.5, 0.0, 0.5, 0.0, 0.5]
    # foot_phases_origin = [0.0, 0.5, 0.5, 0.5, 0.0, 0.0]
    swing_height = 0.1
    swing_foot_track_ema = 0.25

    # Randomization
    nominal_foothold_base_sigma = 0.02
    nominal_base_height_sigma = 0.02
    nominal_swing_height_sigma = 0.05
    # nominal_foothold_base_sigma = 0.1
    # nominal_base_height_sigma = 0.03
    # nominal_swing_height_sigma = 0.1
    min_base_height = 0.16
    min_swing_height = 0.02

    # Reward
    reward_sigma = 0.25

    def __init__(self) -> None:
        self.nominal_foothold_base = [self.nominal_foothold_base_origin_index[i] for i in self.foothold_index_remap]
        self.foot_phases = [self.foot_phases_origin[i] for i in self.foothold_index_remap]
        # self.foot_phases = self.foot_phases_origin
        self.foot_num = len(self.nominal_foothold_base)


class SimpleRaibertPlanner(object):
    def __init__(self, num_envs, device, planner_cfg: SimpleRaibertPlannerConfig) -> None:
        self.num_envs = num_envs
        self.device = device
        self.planner_cfg = planner_cfg

        self.base_x_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.base_y_vec = to_torch([0., 1., 0.], device=self.device).repeat((self.num_envs, 1))
        self.base_z_vec = to_torch([0., 0., 1.], device=self.device).repeat((self.num_envs, 1))

        self.gait_idx = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_phases = [torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)]*6
        self.nominal_foothold_base_rand = to_torch(self.planner_cfg.nominal_foothold_base, device=self.device).repeat((self.num_envs, 1, 1)) \
            + torch.randn((self.num_envs, self.planner_cfg.foot_num, 3),
                          device=self.device) * self.planner_cfg.nominal_foothold_base_sigma
        self.nominal_base_height_rand = torch.tensor(self.planner_cfg.nominal_base_height, device=self.device).repeat(self.num_envs) \
            + torch.randn(self.num_envs, device=self.device) * self.planner_cfg.nominal_base_height_sigma
        self.nominal_base_height_rand = torch.clamp(self.nominal_base_height_rand, self.planner_cfg.min_base_height)
        self.nominal_swing_height_rand = torch.tensor(self.planner_cfg.swing_height, device=self.device).repeat(self.num_envs) \
            + torch.randn(self.num_envs, device=self.device) * self.planner_cfg.nominal_swing_height_sigma
        self.nominal_swing_height_rand = torch.clamp(self.nominal_swing_height_rand, self.planner_cfg.min_swing_height)
        self.last_contacts = torch.zeros(self.num_envs, self.planner_cfg.foot_num,
                                         dtype=torch.bool, device=self.device, requires_grad=False)

        self.is_init = False

    def _get_next_st_mid_duration(self):
        return [torch.remainder(1.75 - phase, 1.0)*self.planner_cfg.gait_period for phase in self.gait_phases]

    def init(self, base_pos: torch.Tensor, base_quat: torch.Tensor):
        # Base Pose Init
        self.base_pos = base_pos.clone()
        # FIXME: Temporarily let base at nominal height
        self.base_pos[:, 2] = self.nominal_base_height_rand
        self.base_quat = base_quat.clone()
        # FIXME: Temporarily let base only rotate around z axis
        base_x_world = quat_apply(self.base_quat, self.base_x_vec)
        heading_angle = torch.atan2(base_x_world[:, 1], base_x_world[:, 0])
        self.base_quat = quat_from_angle_axis(heading_angle, self.base_z_vec)

        # Foot Pos Init
        self.foot_pos = self.nominal_foothold_base_rand.clone()
        self.foot_is_swing = [0.0]*6
        # self.foot_pos = quat_rotate(self.base_quat, self.foot_pos.view(-1, 3)).view(self.num_envs, -1, 3)
        for i in range(self.foot_pos.shape[1]):
            self.foot_pos[:, i] = quat_rotate(self.base_quat, self.foot_pos[:, i])
            self.foot_pos[:, i] += self.base_pos
        self.is_init = True

    def reset_idx(self, base_pos: torch.Tensor, base_quat: torch.Tensor, env_ids):
        # Base Pose Reset
        self.base_pos[env_ids] = base_pos[env_ids]
        # Update nominal base height
        self.nominal_base_height_rand[env_ids] = torch.tensor(self.planner_cfg.nominal_base_height, device=self.device).repeat(len(env_ids)) \
            + torch.randn(len(env_ids), device=self.device) * self.planner_cfg.nominal_base_height_sigma
        self.base_pos[env_ids, 2] = self.nominal_base_height_rand[env_ids]

        self.base_quat[env_ids] = base_quat[env_ids]
        self.base_x_world = quat_apply(self.base_quat, self.base_x_vec)
        heading_angle = torch.atan2(self.base_x_world[env_ids, 1], self.base_x_world[env_ids, 0])
        self.base_quat[env_ids] = quat_from_angle_axis(heading_angle, self.base_z_vec[env_ids])

        # Foot Pos Reset
        # Update nominal foothold base
        self.nominal_foothold_base_rand[env_ids] = to_torch(self.planner_cfg.nominal_foothold_base, device=self.device).repeat((len(env_ids), 1, 1)) \
            + torch.randn((len(env_ids), self.planner_cfg.foot_num, 3),
                          device=self.device) * self.planner_cfg.nominal_foothold_base_sigma
        self.foot_pos[env_ids] = self.nominal_foothold_base_rand[env_ids]
        for i in range(self.foot_pos.shape[1]):
            self.foot_pos[env_ids, i] = quat_rotate(self.base_quat[env_ids], self.foot_pos[env_ids, i])
            self.foot_pos[env_ids, i] += self.base_pos[env_ids]

    def step(self, command):
        # command: [lin_vel_x, lin_vel_y, ang_vel_yaw] Tensor
        base_x_world = quat_apply(self.base_quat, self.base_x_vec)
        base_y_world = quat_apply(self.base_quat, self.base_y_vec)
        # Calculate base pos at the mid of next stance phase
        duration_st_mid = self._get_next_st_mid_duration()
        base_pos_st_mid = [self.base_pos.clone() for i in range(self.planner_cfg.foot_num)]
        base_quat_st_mid = [self.base_quat.clone() for i in range(self.planner_cfg.foot_num)]
        angle_st_mid = [command[:, 2] * dur for dur in duration_st_mid]
        for i in range(self.planner_cfg.foot_num):
            base_pos_st_mid[i] += base_x_world * command[:, :1] * duration_st_mid[i].unsqueeze(-1) \
                + base_y_world * command[:, 1:2] * duration_st_mid[i].unsqueeze(-1)
            base_quat_st_mid[i] = quat_mul(quat_from_angle_axis(angle_st_mid[i], self.base_z_vec), base_quat_st_mid[i])

        # Step base pose
        angle = command[:, 2] * self.planner_cfg.dt
        self.base_quat = quat_mul(quat_from_angle_axis(angle, self.base_z_vec), self.base_quat)
        self.base_pos += base_x_world * command[:, :1] * self.planner_cfg.dt \
            + base_y_world * command[:, 1:2] * self.planner_cfg.dt

        # Step foot pos
        self.gait_idx = torch.remainder(self.gait_idx + self.planner_cfg.dt / self.planner_cfg.gait_period, 1.0)
        self.gait_phases = [torch.remainder(self.gait_idx + phase, 1.0) for phase in self.planner_cfg.foot_phases]
        self.foot_is_swing = [1.0 if phase[0] < 0.5 else 0 for phase in self.gait_phases]  # Temporarily sync all num_envs
        nominal_footpos = self.nominal_foothold_base_rand.clone()
        for i in range(self.nominal_foothold_base_rand.shape[1]):
            nominal_footpos[:, i] = quat_rotate(base_quat_st_mid[i], nominal_footpos[:, i])
            nominal_footpos[:, i] += base_pos_st_mid[i]
        for i, is_swing in enumerate(self.foot_is_swing):
            if is_swing:
                self.foot_pos[:, i, :2] = nominal_footpos[:, i, :2]*self.planner_cfg.swing_foot_track_ema\
                    + self.foot_pos[:, i, :2]*(1-self.planner_cfg.swing_foot_track_ema)
                self.foot_pos[:, i, 2] = sin_swing_traj(self.nominal_swing_height_rand, self.gait_phases[i])
            else:
                self.foot_pos[:, i, 2] = 0.0

    def get_obs_tensor(self, base_pos_real, base_quat_real):
        exp_base_pos_rel = quat_rotate_inverse(base_quat_real, self.base_pos - base_pos_real)
        exp_base_quat_rel = quat_mul(quat_conjugate(base_quat_real), self.base_quat)
        exp_foot_pos_rel = torch.zeros_like(self.foot_pos)
        exp_foot_support = torch.tensor([1.0 if phase[0] > 0.5 else 0.0 for phase in self.gait_phases],
                                        device=self.device).repeat(self.num_envs, 1)
        for i in range(self.foot_pos.shape[1]):
            exp_foot_pos_rel[:, i] = quat_rotate_inverse(base_quat_real, self.foot_pos[:, i] - base_pos_real)
        return torch.cat([exp_base_pos_rel, exp_base_quat_rel, exp_foot_pos_rel.view(self.num_envs, -1),
                          exp_foot_support], dim=-1)

    # Reward as penalty
    def penalty_base_pos_track(self, base_pos_real):
        base_pos_diff = self.base_pos - base_pos_real
        return torch.norm(base_pos_diff, dim=-1)

    def penalty_base_quat_track(self, base_quat_real):
        base_quat_diff = quat_mul(base_quat_real, quat_conjugate(self.base_quat))
        return torch.norm(base_quat_diff[:, :3], dim=-1)

    def penalty_foot_pos_track(self, foot_positions):
        foot_pos_diff = self.foot_pos - foot_positions
        return torch.sum(torch.norm(foot_pos_diff, dim=-1), dim=-1)

    def penalty_foot_pos_track_z(self, foot_positions):
        foot_pos_diff = self.foot_pos - foot_positions
        return torch.sum(torch.norm(foot_pos_diff[:, :, 2:], dim=-1), dim=-1)

    def penalty_foot_swing_contact(self, foot_contact_force, feet_indices):
        # penalize foot contact force when foot is in swing phase
        contact = foot_contact_force[:, feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        # Force norm
        # foot_contact_force_norm = torch.norm(foot_contact_force[:, self.feet_indices, :], dim=-1)
        # foot_contact_force_penalty = torch.zeros(self.num_envs, device=self.device)
        # for i, is_swing in enumerate(self.foot_is_swing):
        #     foot_contact_force_penalty += foot_contact_force_norm[:, i] * is_swing
        # return foot_contact_force_penalty
        # Contact
        penalty = torch.zeros(self.num_envs, device=self.device)
        for i, is_swing in enumerate(self.foot_is_swing):
            penalty += contact_filt[:, i] * is_swing
        return penalty

    # Reward as reward
    def reward_base_pos_track(self, base_pos_real):
        base_pos_diff = self.base_pos - base_pos_real
        return torch.exp(-torch.norm(base_pos_diff, dim=-1) / self.planner_cfg.reward_sigma)

    def reward_base_quat_track(self, base_quat_real):
        base_quat_diff = quat_mul(base_quat_real, quat_conjugate(self.base_quat))
        return torch.exp(-torch.norm(base_quat_diff[:, :3], dim=-1) / self.planner_cfg.reward_sigma)

    def reward_foot_pos_track(self, foot_positions):
        foot_pos_diff = self.foot_pos - foot_positions
        return torch.sum(torch.exp(-torch.norm(foot_pos_diff, dim=-1) / self.planner_cfg.reward_sigma), dim=-1)


class RaibertPlannerConfig:

    # Environment & Simulation
    dt = 0.02

    # Nominal Foothold in Base Frame
    nominal_y_shift = 0.06
    nominal_x_shift = 0.0
    nominal_foothold_base_origin_index = [
        [0.354 + nominal_x_shift, -0.28 - nominal_y_shift, -0.28],
        [0.054 + nominal_x_shift, -0.34 - nominal_y_shift, -0.28],
        [-0.354 + nominal_x_shift, -0.28 - nominal_y_shift, -0.28],
        [0.354 + nominal_x_shift, 0.28 + nominal_y_shift, -0.28],
        [0.054 + nominal_x_shift, 0.34 + nominal_y_shift, -0.28],
        [-0.354 + nominal_x_shift, 0.28 + nominal_y_shift, -0.28],
    ]
    # Origin Leg order: RF, RM, RB, LF, LM, LB
    # URDF alphabet order: LB, LF, LM, RB, RF, RM
    foothold_index_remap = [5, 3, 4, 2, 0, 1]
    # Randomization
    nominal_foothold_base_sigma = 0.08
    foothold_target_update_interval = 0.5
    foothold_target_track_kp = 1.5
    foothold_max_track_vel = 2.0

    # Base Pose
    base_height_bound = [0.16, 0.40]
    # Base Pose Shift Bound
    base_xshift_bound = [-0.1, 0.1]
    base_yshift_bound = [-0.1, 0.1]
    base_yaw_bound = [-0.5, 0.5]
    base_pitch_bound = [-0.3, 0.3]
    base_roll_bound = [-0.8, 0.8]
    # Randomization
    basepose_target_update_interval = 0.5
    basepose_target_track_kp = 1.5
    basepose_max_track_vel = 1.0

    # Gait
    gait_period = 0.5
    foot_phases_origin = [0.0, 0.5, 0.0, 0.5, 0.0, 0.5]
    enable_rand_gait = False  # Randomize foot selection (2-gait) for each env
    swing_foot_track_ema = 0.25

    # Swing Trajectory
    swing_traj_liftv_bound = [[-0.2, -0.2, 0.0],
                              [0.2, 0.2, 0.3]]
    swing_traj_touchv_bound = [[-0.2, -0.2, 0.0],
                               [0.2, 0.2, -0.3]]

    # Reward
    reward_sigma = 0.25

    def __init__(self) -> None:
        self.base_rand_bound = np.array([self.base_xshift_bound, self.base_yshift_bound, self.base_height_bound,
                                         self.base_yaw_bound, self.base_pitch_bound, self.base_roll_bound]).T
        self.nominal_foothold_base_rand_bound = \
            np.concatenate((np.array([self.nominal_foothold_base_origin_index[i]
                                      for i in self.foothold_index_remap]).reshape(1, -1),
                            np.array([self.nominal_foothold_base_sigma]*18).reshape(1, -1)), axis=0)

        self.foot_phases = [self.foot_phases_origin[i] for i in self.foothold_index_remap]
        self.foot_num = len(self.foothold_index_remap)


class RaibertPlanner:
    def __init__(self, num_envs, device, planner_cfg: RaibertPlannerConfig) -> None:
        self.num_envs = num_envs
        self.device = device
        self.planner_cfg = planner_cfg

        self.base_x_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.base_y_vec = to_torch([0., 1., 0.], device=self.device).repeat((self.num_envs, 1))
        self.base_z_vec = to_torch([0., 0., 1.], device=self.device).repeat((self.num_envs, 1))
        self.base_x_world = self.base_x_vec.clone()
        self.base_y_world = self.base_y_vec.clone()

        self.gait_idx = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_phases = [torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)]*6

        self.last_contacts = torch.zeros(self.num_envs, self.planner_cfg.foot_num,
                                         dtype=torch.bool, device=self.device, requires_grad=False)

        # Rand Vec: [x_shift, y_shift, height, yaw, pitch, roll]
        self.base_pose_randwalk = RandomWalker(torch.tensor(self.planner_cfg.base_rand_bound, device=self.device),
                                               num_envs=self.num_envs,
                                               target_update_interval=self.planner_cfg.basepose_target_update_interval,
                                               target_track_kp=self.planner_cfg.basepose_target_track_kp,
                                               max_track_vel=self.planner_cfg.basepose_max_track_vel,
                                               distribution_type='uniform')
        self.foothold_base_randwalk = \
            RandomWalker(torch.tensor(self.planner_cfg.nominal_foothold_base_rand_bound, device=self.device),
                         num_envs=self.num_envs,
                         target_update_interval=self.planner_cfg.foothold_target_update_interval,
                         target_track_kp=self.planner_cfg.foothold_target_track_kp,
                         max_track_vel=self.planner_cfg.foothold_max_track_vel,
                         distribution_type='normal')

        self.is_init = False

    def _get_unsqueeze_rand_foothold_base(self):
        return self.foothold_base_randwalk.positions.view(self.num_envs, self.planner_cfg.foot_num, 3).clone()

    def _get_base_quat_shift_from_rand(self, base_quat=None):
        if base_quat is None:
            base_quat = self.base_quat
        return quat_mul(base_quat, ypr_to_quat(self.base_pose_randwalk.positions[:, 3],
                                               self.base_pose_randwalk.positions[:, 4],
                                               self.base_pose_randwalk.positions[:, 5]))

    def _get_base_pos_shift_from_rand(self, base_pos=None):
        if base_pos is None:
            base_pos = self.base_pos
        return base_pos + self.base_x_world * self.base_pose_randwalk.positions[:, 0].unsqueeze(-1) \
            + self.base_y_world * self.base_pose_randwalk.positions[:, 1].unsqueeze(-1)

    def _get_base_height_from_rand(self):
        return self.base_pose_randwalk.positions[:, 2]

    def _get_next_st_mid_duration(self):
        return [torch.remainder(1.75 - phase, 1.0)*self.planner_cfg.gait_period for phase in self.gait_phases]

    def init(self, base_pos: torch.Tensor, base_quat: torch.Tensor):
        # Base Pose Init
        self.base_pos = base_pos.clone()
        self.base_pos[:, 2] = self._get_base_height_from_rand()
        self.base_pos_shift = self._get_base_pos_shift_from_rand()
        self.base_quat = base_quat.clone()
        self.base_x_world = quat_apply(self.base_quat, self.base_x_vec)
        heading_angle = torch.atan2(self.base_x_world[:, 1], self.base_x_world[:, 0])
        self.base_quat = quat_from_angle_axis(heading_angle, self.base_z_vec)
        self.base_quat_shift = self._get_base_quat_shift_from_rand()

        # Foot Pos Init
        self.foot_pos = self._get_unsqueeze_rand_foothold_base()
        self.foot_is_swing = [0.0]*6
        for i in range(self.planner_cfg.foot_num):
            self.foot_pos[:, i] = quat_rotate(self.base_quat, self.foot_pos[:, i])
            self.foot_pos[:, i] += self.base_pos
        self.is_init = True

    def reset_idx(self, base_pos: torch.Tensor, base_quat: torch.Tensor, env_ids):
        # Base Pose Reset
        self.base_pos[env_ids] = base_pos[env_ids]
        self.base_pos[env_ids, 2] = self._get_base_height_from_rand()[env_ids]
        self.base_pos_shift[env_ids] = self._get_base_pos_shift_from_rand()[env_ids]

        self.base_quat[env_ids] = base_quat[env_ids]
        self.base_x_world = quat_apply(self.base_quat, self.base_x_vec)
        heading_angle = torch.atan2(self.base_x_world[env_ids, 1], self.base_x_world[env_ids, 0])
        self.base_quat[env_ids] = quat_from_angle_axis(heading_angle, self.base_z_vec[env_ids])
        self.base_quat_shift[env_ids] = self._get_base_quat_shift_from_rand()[env_ids]

        # Foot Pos Reset
        self.foot_pos[env_ids] = self._get_unsqueeze_rand_foothold_base()[env_ids]
        for i in range(self.foot_pos.shape[1]):
            self.foot_pos[env_ids, i] = quat_rotate(self.base_quat[env_ids], self.foot_pos[env_ids, i])
            self.foot_pos[env_ids, i] += self.base_pos[env_ids]

    def step(self, command):
        # Step random walkers
        self.base_pose_randwalk.step(self.planner_cfg.dt)
        self.foothold_base_randwalk.step(self.planner_cfg.dt)

        # command: [lin_vel_x, lin_vel_y, ang_vel_yaw] Tensor
        self.base_x_world = quat_apply(self.base_quat, self.base_x_vec)
        self.base_y_world = quat_apply(self.base_quat, self.base_y_vec)
        # Calculate base pos at the mid of next stance phase
        duration_st_mid = self._get_next_st_mid_duration()
        base_pos_st_mid = [self.base_pos.clone() for i in range(self.planner_cfg.foot_num)]
        base_quat_st_mid = [self.base_quat.clone() for i in range(self.planner_cfg.foot_num)]
        angle_st_mid = [command[:, 2] * dur for dur in duration_st_mid]
        for i in range(self.planner_cfg.foot_num):
            base_pos_st_mid[i] += self.base_x_world * command[:, :1] * duration_st_mid[i].unsqueeze(-1) \
                + self.base_y_world * command[:, 1:2] * duration_st_mid[i].unsqueeze(-1)
            # base_pos_st_mid[i] = self._get_base_pos_shift_from_rand(base_pos_st_mid[i])
            base_quat_st_mid[i] = quat_mul(quat_from_angle_axis(angle_st_mid[i], self.base_z_vec), base_quat_st_mid[i])
            # base_quat_st_mid[i] = self._get_base_quat_shift_from_rand(base_quat_st_mid[i])

        # Step base pose
        angle = command[:, 2] * self.planner_cfg.dt
        self.base_quat = quat_mul(quat_from_angle_axis(angle, self.base_z_vec), self.base_quat)
        self.base_quat_shift = self._get_base_quat_shift_from_rand()
        self.base_pos += self.base_x_world * command[:, :1] * self.planner_cfg.dt \
            + self.base_y_world * command[:, 1:2] * self.planner_cfg.dt
        self.base_pos[:, 2] = self._get_base_height_from_rand()
        self.base_pos_shift = self._get_base_pos_shift_from_rand()

        # Step foot pos
        self.gait_idx = torch.remainder(self.gait_idx + self.planner_cfg.dt / self.planner_cfg.gait_period, 1.0)
        self.gait_phases = [torch.remainder(self.gait_idx + phase, 1.0) for phase in self.planner_cfg.foot_phases]
        self.foot_is_swing = [1.0 if phase[0] < 0.5 else 0 for phase in self.gait_phases]  # Temporarily sync all num_envs
        nominal_footpos = self._get_unsqueeze_rand_foothold_base()
        for i in range(self.planner_cfg.foot_num):
            nominal_footpos[:, i] = quat_rotate(base_quat_st_mid[i], nominal_footpos[:, i])
            nominal_footpos[:, i] += base_pos_st_mid[i]
        for i, is_swing in enumerate(self.foot_is_swing):
            if is_swing:
                self.foot_pos[:, i, :2] = nominal_footpos[:, i, :2]*self.planner_cfg.swing_foot_track_ema \
                    + self.foot_pos[:, i, :2]*(1-self.planner_cfg.swing_foot_track_ema)
                # FIXME: add rand hermite swing trajectory
                self.foot_pos[:, i, 2] = sin_swing_traj(0.1, self.gait_phases[i])
            else:
                self.foot_pos[:, i, 2] = 0.0

    def get_obs_tensor(self, base_pos_real, base_quat_real):
        exp_base_pos_rel = quat_rotate_inverse(base_quat_real, self.base_pos_shift - base_pos_real)
        exp_base_quat_rel = quat_mul(quat_conjugate(base_quat_real), self.base_quat_shift)
        exp_foot_pos_rel = torch.zeros_like(self.foot_pos)
        exp_foot_support = torch.tensor([1.0 if phase[0] > 0.5 else 0.0 for phase in self.gait_phases],
                                        device=self.device).repeat(self.num_envs, 1)
        for i in range(self.foot_pos.shape[1]):
            exp_foot_pos_rel[:, i] = quat_rotate_inverse(base_quat_real, self.foot_pos[:, i] - base_pos_real)
        return torch.cat([exp_base_pos_rel, exp_base_quat_rel, exp_foot_pos_rel.view(self.num_envs, -1),
                          exp_foot_support], dim=-1)

    # Reward as penalty
    def penalty_base_pos_track(self, base_pos_real):
        base_pos_diff = self.base_pos_shift - base_pos_real
        return torch.norm(base_pos_diff, dim=-1)

    def penalty_base_quat_track(self, base_quat_real):
        base_quat_diff = quat_mul(base_quat_real, quat_conjugate(self.base_quat_shift))
        return torch.norm(base_quat_diff[:, :3], dim=-1)

    def penalty_foot_pos_track(self, foot_positions):
        foot_pos_diff = self.foot_pos - foot_positions
        return torch.sum(torch.norm(foot_pos_diff, dim=-1), dim=-1)

    def penalty_foot_pos_track_z(self, foot_positions):
        foot_pos_diff = self.foot_pos - foot_positions
        return torch.sum(torch.norm(foot_pos_diff[:, :, 2:], dim=-1), dim=-1)

    def penalty_foot_swing_contact(self, foot_contact_force, feet_indices):
        # penalize foot contact force when foot is in swing phase
        contact = foot_contact_force[:, feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        # Force norm
        # foot_contact_force_norm = torch.norm(foot_contact_force[:, self.feet_indices, :], dim=-1)
        # foot_contact_force_penalty = torch.zeros(self.num_envs, device=self.device)
        # for i, is_swing in enumerate(self.foot_is_swing):
        #     foot_contact_force_penalty += foot_contact_force_norm[:, i] * is_swing
        # return foot_contact_force_penalty
        # Contact
        penalty = torch.zeros(self.num_envs, device=self.device)
        for i, is_swing in enumerate(self.foot_is_swing):
            penalty += contact_filt[:, i] * is_swing
        return penalty

    # Reward as reward
    def reward_base_pos_track(self, base_pos_real):
        base_pos_diff = self.base_pos_shift - base_pos_real
        return torch.exp(-torch.norm(base_pos_diff, dim=-1) / self.planner_cfg.reward_sigma)

    def reward_base_quat_track(self, base_quat_real):
        base_quat_diff = quat_mul(base_quat_real, quat_conjugate(self.base_quat_shift))
        return torch.exp(-torch.norm(base_quat_diff[:, :3], dim=-1) / self.planner_cfg.reward_sigma)

    def reward_foot_pos_track(self, foot_positions):
        foot_pos_diff = self.foot_pos - foot_positions
        return torch.sum(torch.exp(-torch.norm(foot_pos_diff, dim=-1) / self.planner_cfg.reward_sigma), dim=-1)
