'''
Author: Raymon Yip 2205929492@qq.com
Date: 2025-03-12 13:53:01
Description: file content
FilePath: /PredictiveDiffusionPlanner_Dev/legged_gym_cmp/legged_gym/legged_gym/utils/gait_scheduler.py
LastEditTime: 2025-06-18 20:08:56
LastEditors: RCAMC-4090
'''
from typing import Optional
import torch


def sin_swing_traj(swing_height, phase: torch.Tensor):
    # if 0 < phase < 0.5: sin swing trajectory, else 0
    return torch.where(phase < 0.5, swing_height * torch.sin(2 * torch.pi * phase), torch.zeros_like(phase))


class GaitSchedulerCfg(object):
    period = 1.0
    duty = 0.5
    # foot_phases = [0.0, 0.5, 0.0, 0.5]
    foot_phases = [0.0, 0.5, 0.0, 0.5, 0.0, 0.5]
    dt = 0.02
    swing_height = 0.04
    track_sigma = 0.25


class GaitScheduler(object):
    def __init__(self,
                 height_samples: Optional[torch.Tensor],
                 base_quat: torch.Tensor,
                 base_lin_vel: torch.Tensor,
                 base_ang_vel: torch.Tensor,
                 projected_gravity: torch.Tensor,
                 dof_pos: torch.Tensor,
                 dof_vel: torch.Tensor,
                 foot_pos: torch.Tensor,
                 foot_vel: torch.Tensor,
                 num_envs, device,
                 gait_cfg: GaitSchedulerCfg = GaitSchedulerCfg(),) -> None:
        # Terrain Reference
        self.height_samples = height_samples
        # Robot State Reference
        self.base_quat = base_quat
        self.base_lin_vel = base_lin_vel
        self.base_ang_vel = base_ang_vel
        self.projected_gravity = projected_gravity
        self.dof_pos = dof_pos
        self.dof_vel = dof_vel
        self.foot_pos = foot_pos
        self.foot_vel = foot_vel

        # Other
        self.num_envs = num_envs
        self.device = device
        self.gait_cfg = gait_cfg

        # Gait
        # Gait index between 0 and 1
        self.gait_idx = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_phases = []

    def step(self, foot_pos: torch.Tensor, foot_vel: torch.Tensor, cmd=None, t=None):
        self.cmd = cmd
        if t is not None:
            self.gait_idx = torch.remainder(t / self.gait_cfg.period * torch.ones(self.num_envs,
                                            device=self.device, dtype=torch.float, requires_grad=False), 1.0)
        else:
            self.gait_idx = torch.remainder(self.gait_idx + self.gait_cfg.dt / self.gait_cfg.period, 1.0)
        self.gait_phases = [torch.remainder(self.gait_idx + phase, 1.0) for phase in self.gait_cfg.foot_phases]
        self.foot_pos = foot_pos
        self.foot_vel = foot_vel

    def reward_foot_z_track(self):
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        for i in range(len(self.gait_phases)):
            foot_pos = self.foot_pos[:, i]
            foot_pos_target = sin_swing_traj(self.gait_cfg.swing_height, self.gait_phases[i])
            foot_pos_z_diff = foot_pos_target - foot_pos[:, 2]
            reward += torch.square(foot_pos_z_diff)
        return reward

    # def reward_foot_z_track(self):
    #     reward = torch.ones(self.num_envs, device=self.device, dtype=torch.float)
    #     for i in range(len(self.gait_phases)):
    #         foot_pos = self.foot_pos[:, i]
    #         foot_pos_target = sin_swing_traj(self.gait_cfg.swing_height, self.gait_phases[i])
    #         foot_pos_z_diff = foot_pos_target - foot_pos[:, 2]
    #         reward += torch.where(foot_pos_target > 0.03,
    #                               torch.exp(-torch.square(foot_pos_z_diff)/self.gait_cfg.track_sigma),
    #                               torch.zeros_like(foot_pos_z_diff))
    #         # print(i, foot_pos)
    #     return reward


class AsyncGaitSchedulerCfg(object):
    # Same tag should keep same motion
    dof_names = ['LB_HAA', 'LB_HFE', 'LB_KFE',
                 'LF_HAA', 'LF_HFE', 'LF_KFE',
                 'LM_HAA', 'LM_HFE', 'LM_KFE',
                 'RB_HAA', 'RB_HFE', 'RB_KFE',
                 'RF_HAA', 'RF_HFE', 'RF_KFE',
                 'RM_HAA', 'RM_HFE', 'RM_KFE']

    dof_align_sets = [['RF_HFE', 'RB_HFE', 'LM_HFE'],
                      ['LF_HFE', 'LB_HFE', 'RM_HFE'],
                      ['RF_KFE', 'RB_KFE', 'LM_KFE'],
                      ['LF_KFE', 'LB_KFE', 'RM_KFE'],]
    dof_nominal_pos = [0.0, 1.0, 1.0]*6  # HAA, HFE, KFE
    dof_nominal_pos_weight = [1.0, 1.0, 3.0]*6  # HAA, HFE, KFE

    foot_names = ['LB_FOOT', 'LF_FOOT', 'LM_FOOT', 'RB_FOOT', 'RF_FOOT', 'RM_FOOT']
    foot_z_align_sets = [['RF_FOOT', 'RB_FOOT', 'LM_FOOT'],
                         ['LF_FOOT', 'LB_FOOT', 'RM_FOOT'],]

    def __init__(self) -> None:
        self.dof_align_sets_idx = [[self.dof_names.index(dof) for dof in dof_set] for dof_set in self.dof_align_sets]
        self.foot_z_align_sets_idx = [[self.foot_names.index(foot) for foot in foot_set] for foot_set in self.foot_z_align_sets]


class AsyncGaitScheduler(object):
    def __init__(self,
                 height_samples: Optional[torch.Tensor],
                 base_quat: torch.Tensor,
                 base_lin_vel: torch.Tensor,
                 base_ang_vel: torch.Tensor,
                 projected_gravity: torch.Tensor,
                 dof_pos: torch.Tensor,
                 dof_vel: torch.Tensor,
                 foot_pos: torch.Tensor,
                 foot_vel: torch.Tensor,
                 num_envs, device,
                 gait_cfg: AsyncGaitSchedulerCfg = AsyncGaitSchedulerCfg(),) -> None:
        # Terrain Reference
        self.height_samples = height_samples
        # Robot State Reference (Updated at every step)
        self.base_quat = base_quat
        self.base_lin_vel = base_lin_vel
        self.base_ang_vel = base_ang_vel
        self.projected_gravity = projected_gravity
        self.dof_pos = dof_pos
        self.dof_vel = dof_vel
        self.foot_pos = foot_pos
        self.foot_vel = foot_vel

        # Other
        self.num_envs = num_envs
        self.device = device
        self.gait_cfg = gait_cfg

    def reward_dof_align(self):
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        for dof_set_idx in self.gait_cfg.dof_align_sets_idx:
            sel_dof_pos = self.dof_pos[:, dof_set_idx]
            reward += torch.std(sel_dof_pos, dim=1)
        return reward

    def reward_dof_nominal_pos(self):
        dof_nomianl_pos_mat = torch.tensor(self.gait_cfg.dof_nominal_pos,
                                           device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        dof_err = torch.square(self.dof_pos - dof_nomianl_pos_mat)
        dof_err_weight = torch.tensor(self.gait_cfg.dof_nominal_pos_weight,
                                      device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        weighted_err = dof_err * dof_err_weight
        return torch.sum(weighted_err, dim=1)

    def reward_foot_z_align(self):
        """ z align is only for flat env """
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        for foot_set_idx in self.gait_cfg.foot_z_align_sets_idx:
            sel_foot_pos_z = self.foot_pos[:, foot_set_idx, 2]
            reward += torch.std(sel_foot_pos_z, dim=1)
        return reward

    def reward_contact_state_align(self):
        pass


class MultiGaitSchedulerCfg(object):
    foot_num = 6
    gait_patterns = []


class MultiGaitScheduler(object):
    pass


if __name__ == "__main__":
    print(AsyncGaitSchedulerCfg().dof_align_sets_idx)
