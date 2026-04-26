from typing import Dict, Tuple
import csv
import os
from datetime import datetime

import torch
from torch import Tensor
import numpy as np

from legged_gym.envs.el_4090.spider_nomal.el_4090 import EL_4090
from legged_gym.utils.math_utils import quat_rotate_inverse
from legged_gym.utils.atacom import ATACOMSafetyLayer
from .el_4090_safe_config import El4090SafeCfg


class EL_4090_Safe(EL_4090):
    """EL_4090 + ATACOM 安全层。

    step(action) 接收 RL 名义动作，经 ATACOM 转换为安全动作后再传给父类仿真。

    运行时状态布局（S=58）：
        [0 :18)   dof_pos          关节位置
        [18:36)   dof_vel          关节速度
        [36:54)   torques          关节实际力矩（self.torques，上一步）
        [54:57)   base_euler       机身欧拉角（roll, pitch, yaw），ZYX 旋转顺序
        [57:58)   base_pos_z       机身高度

    性能说明：
        - ATACOM forward 的 info 字典不再含 .item()，不触发 GPU 同步
        - 标量聚合由 ATACOMSafetyLayer.compute_info_scalars(info) 负责
        - log_interval 控制聚合频率（默认每 100 步一次），大幅减少同步开销
    """

    _BASE_OBS_DIM = 66
    cfg: El4090SafeCfg

    def __init__(self, cfg: El4090SafeCfg, sim_params, physics_engine, sim_device, headless,
                 task_name="el_4090_safe"):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,
                         task_name=task_name)


        safety_cfg = getattr(self.cfg, 'safety', El4090SafeCfg.safety)

        def _to_tensor_18(x, default):
            val = getattr(safety_cfg, x, default)
            if isinstance(val, (int, float)):
                val = [val] * 18
            return torch.tensor(val, device=self.device, dtype=torch.float32)

        def _to_tensor_3(x, default):
            val = getattr(safety_cfg, x, default)
            if isinstance(val, (int, float)):
                val = [val] * 3
            return torch.tensor(val, device=self.device, dtype=torch.float32)

        robot_params: Dict = {
            'q_max'  : _to_tensor_18('q_max',   [2.95]  * 18),
            'q_min'  : _to_tensor_18('q_min',   [-2.95] * 18),
            'dq_max' : _to_tensor_18('dq_max',  [14.2]  * 18),
            'tau_max': _to_tensor_18('tau_max',  [76]    * 18),
            'phi_max': _to_tensor_3( 'phi_max',  [0.14, 0.14, 3.14]),
            'z_max'  : float(getattr(safety_cfg, 'z_max',  0.8)),
            'z_min'  : float(getattr(safety_cfg, 'z_min',  0.2)),
        }

        # 初始化 ATACOM 安全层
        self.atacom = ATACOMSafetyLayer(
            robot_params=robot_params,
            lambda_retract=float(getattr(safety_cfg, 'lambda_retract', 1.0)),
            beta=float(getattr(safety_cfg, 'beta', 2.0)),
            dt=float(getattr(safety_cfg, 'dt', self.dt)),
            debug_mode=bool(getattr(safety_cfg, 'debug_mode', False)),
            debug_level=getattr(safety_cfg, 'debug_level', 'basic'),
            debug_interval=int(getattr(safety_cfg, 'debug_interval', 100)),
        )

        self._atacom_enabled      = bool(getattr(safety_cfg, 'enable_atacom',        True))
        self._atacom_clip_nominal = bool(getattr(safety_cfg, 'clip_nominal_actions',  True))
        self._atacom_warmup       = int(getattr(safety_cfg,  'warmup_steps',          0))
        self._atacom_log_info     = bool(getattr(safety_cfg, 'log_info',              False))

        # 约束违反记录配置
        self._record_violation = bool(getattr(safety_cfg, 'record_violation', True))
        self._record_interval = max(1, int(getattr(safety_cfg, 'record_violation_interval', 1)))
        self._record_violation_detail = bool(getattr(safety_cfg, 'record_violation_detail', True))
        self._record_violation_topk = max(1, int(getattr(safety_cfg, 'record_violation_topk', 5)))
        self._violation_global_max = 0.0
        self._violation_csv_file = None
        self._violation_csv_writer = None
        self._constraint_names = self._build_constraint_names()

        if self._record_violation:
            self._init_violation_recorder()

    def _init_violation_recorder(self):
        """初始化约束违反记录器，输出到当前文件同目录的 data/ 下。"""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'constraint_violation_{timestamp}.csv'
        file_path = os.path.join(data_dir, file_name)

        self._violation_csv_file = open(file_path, 'w', newline='')
        self._violation_csv_writer = csv.writer(self._violation_csv_file)
        header = [
            'common_step',
            'atacom_step',
            'constraint_violation',
            'max_violation',
            'global_max_violation',
            'safe_ratio',
        ]
        if self._record_violation_detail:
            header.extend([
                'violated_constraint_count',
                'violated_constraint_indices',
                'violated_constraint_names',
                'topk_violations',
            ])
        self._violation_csv_writer.writerow(header)
        self._violation_csv_file.flush()

    def _build_constraint_names(self):
        """按 ATACOM k 向量顺序构造约束名称（共 77 项）。"""
        names = []

        # [0:36) 关节位置限制（上下限交错）
        for j in range(18):
            names.append(f"q[{j}]_upper")
            names.append(f"q[{j}]_lower")

        # [36:54) 关节速度限制
        for j in range(18):
            names.append(f"dq[{j}]_abs")

        # [54:72) 关节力矩限制
        for j in range(18):
            names.append(f"tau[{j}]_abs")

        # [72:74) 机身高度限制
        names.extend(['z_upper', 'z_lower'])

        # [74:77) 机身倾角限制
        names.extend(['phi_roll_abs', 'phi_pitch_abs', 'phi_yaw_abs'])

        return names

    def _build_violation_detail(self, k_viol: torch.Tensor):
        """构造违反约束详情字符串（按当前 step 聚合所有环境）。"""
        # 每个约束在所有环境上的最大违反量
        k_max_per = k_viol.max(dim=0).values
        violated_mask = k_max_per > 0
        violated_indices_tensor = torch.nonzero(violated_mask, as_tuple=False).squeeze(-1)

        if violated_indices_tensor.numel() == 0:
            return 0, '', '', ''

        violated_indices = [int(i) for i in violated_indices_tensor.tolist()]
        violated_names = [self._constraint_names[i] if i < len(self._constraint_names) else f"k[{i}]"
                          for i in violated_indices]

        topk = min(self._record_violation_topk, len(violated_indices))
        top_vals, top_idx = torch.topk(k_max_per, k=topk)
        topk_parts = []
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            if val <= 0:
                continue
            name = self._constraint_names[idx] if idx < len(self._constraint_names) else f"k[{idx}]"
            topk_parts.append(f"{idx}:{name}:{val:.6f}")

        return (
            len(violated_indices),
            '|'.join(str(i) for i in violated_indices),
            '|'.join(violated_names),
            '|'.join(topk_parts),
        )

    def _record_constraint_violation(self, atacom_info: Dict):
        """记录约束违反程度与最大违反值。"""
        if (not self._record_violation
                or self._violation_csv_writer is None
                or (self.common_step_counter % self._record_interval) != 0):
            return

        k = atacom_info['k']
        k_viol = torch.clamp(k, min=0)

        constraint_violation = k_viol.sum(dim=1).mean().item()
        max_violation = k_viol.max().item()
        safe_ratio = (k <= 0).all(dim=1).float().mean().item()

        if max_violation > self._violation_global_max:
            self._violation_global_max = max_violation

        row = [
            int(self.common_step_counter),
            int(atacom_info.get('step', -1)),
            float(constraint_violation),
            float(max_violation),
            float(self._violation_global_max),
            float(safe_ratio),
        ]

        if self._record_violation_detail:
            viol_count, viol_indices, viol_names, topk_viol = self._build_violation_detail(k_viol)
            row.extend([
                int(viol_count),
                viol_indices,
                viol_names,
                topk_viol,
            ])

        self._violation_csv_writer.writerow(row)

        # 降低 IO 开销：每 100 条刷盘一次
        if (self.common_step_counter % (self._record_interval * 100)) == 0:
            self._violation_csv_file.flush()

    def _close_violation_recorder(self):
        if self._violation_csv_file is not None:
            try:
                self._violation_csv_file.flush()
                self._violation_csv_file.close()
            finally:
                self._violation_csv_file = None
                self._violation_csv_writer = None

    def __del__(self):
        self._close_violation_recorder()

    # ------------------------------------------------------------------
    # 状态拼装
    # ------------------------------------------------------------------

    def _build_atacom_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """从仿真 buffer 拼装 ATACOM 运行时状态 s，(num_envs, 58)。

        Returns:
            s            : ATACOM 状态向量 (num_envs, 58)
            ang_vel_base : 机体系角速度 (num_envs, 3)
        """
        num_envs = self.num_envs
        device   = self.device
        s = torch.zeros((num_envs, 58), device=device)

        s[:, 0:18]  = self.dof_pos[:, :18]
        s[:, 18:36] = self.dof_vel[:, :18]

        if hasattr(self, 'torques') and self.torques is not None:
            s[:, 36:54] = self.torques[:, :18]

        base_quat     = self.root_states[:, 3:7]
        ang_vel_world = self.root_states[:, 10:13]
        ang_vel_base  = quat_rotate_inverse(base_quat, ang_vel_world)

        qx, qy, qz, qw = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]

        roll  = torch.atan2(2.0 * (qw * qx + qy * qz),
                            1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
        yaw   = torch.atan2(2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz))

        s[:, 54:57] = torch.stack([roll, pitch, yaw], dim=1)
        s[:, 57]    = self.root_states[:, 2]

        return s, ang_vel_base

    # ------------------------------------------------------------------
    # buffer 初始化
    # ------------------------------------------------------------------

    def _init_buffers(self):
        super()._init_buffers()
        self.u_mu = torch.zeros(
            (self.num_envs, 77), device=self.device, dtype=torch.float32
        )

    # ------------------------------------------------------------------
    # 观测计算
    # ------------------------------------------------------------------

    def _get_noise_scale_vec(self, cfg):
        """覆写父类，强制按父类原始 66 维构造噪声向量。"""
        original_obs_buf = self.obs_buf
        self.obs_buf = torch.zeros(
            (self.num_envs, self._BASE_OBS_DIM),
            device=self.device, dtype=torch.float32
        )
        noise_vec    = super()._get_noise_scale_vec(cfg)
        self.obs_buf = original_obs_buf
        return noise_vec

    def compute_observations(self):
        """在父类观测基础上追加 u_mu（77维）→ obs_buf 共 143 维。"""
        if self.obs_buf.shape[1] != self._BASE_OBS_DIM:
            self.obs_buf = self.obs_buf[:, :self._BASE_OBS_DIM].contiguous()
        super().compute_observations()
        self.obs_buf = torch.cat([self.obs_buf, self.u_mu], dim=-1)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, actions, *args, **kwargs) -> Tuple:
        """ATACOM 安全过滤后转发给父类 step。"""

        if (not self._atacom_enabled
                or self.common_step_counter < self._atacom_warmup):
            return super().step(actions, *args, **kwargs)

        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        else:
            actions = actions.to(self.device)

        if self._atacom_clip_nominal:
            clip_val = getattr(self.cfg.normalization, 'clip_actions', None)
            if clip_val is not None:
                actions = torch.clamp(actions, -clip_val, clip_val)

        if actions.ndim == 1:
            actions = actions.unsqueeze(0).expand(self.num_envs, -1)

        s, ang_vel_base = self._build_atacom_state()
        u_safe, u_mu, atacom_info = self.atacom.forward(s, actions, ang_vel_body=ang_vel_base)

        self.u_mu = u_mu
        self._record_constraint_violation(atacom_info)
        # print(u_mu)

        # 聚合标量（可选，会触发 GPU 同步）
        if self._atacom_log_info:
            if not hasattr(self, 'extras'):
                self.extras = {}
            self.extras['atacom'] = ATACOMSafetyLayer.compute_info_scalars(atacom_info)

        return super().step(u_safe, *args, **kwargs)
    

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            print("command has been updated!")
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
            

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return
        # print(f"Resampling small commands for {len(env_ids)} envs (total reward: {mean_total_reward:.3f}, lin reward: {mean_lin_reward:.3f})")
        
        if self.cfg.commands.small_command_radio:
            small_ratio = 0.01
            small_mask = torch.rand(len(env_ids), device=self.device) < small_ratio
            if not torch.any(small_mask):
                return

            small_env_ids = env_ids[small_mask]
            n_small = len(small_env_ids)

            # 小线速度命令：[-0.1, 0.1]
            self.commands[small_env_ids, 0] = (torch.rand(n_small, device=self.device) * 2.0 - 1.0) * 0.1
            self.commands[small_env_ids, 1] = (torch.rand(n_small, device=self.device) * 2.0 - 1.0) * 0.1
            # 小转向命令：[-0.1, 0.1]
            if self.cfg.commands.heading_command:
                self.commands[small_env_ids, 3] = (torch.rand(n_small, device=self.device) * 2.0 - 1.0) * 0.1
            else:
                self.commands[small_env_ids, 2] = (torch.rand(n_small, device=self.device) * 2.0 - 1.0) * 0.1

            



    def _reward_stand_on_six_legs(self):
        # 低命令下：鼓励六条腿全部着地

        lin_cmd_small = torch.norm(self.commands[:, :2], dim=1) < self.speed_min
        if self.cfg.commands.heading_command:
            yaw_or_heading_small = torch.abs(self.commands[:, 3]) < self.speed_min
        else:
            yaw_or_heading_small = torch.abs(self.commands[:, 2]) < self.speed_min
        small_command_mask = torch.logical_and(lin_cmd_small, yaw_or_heading_small)

        # 足端接触（法向力阈值 1N）
        foot_contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        num_feet_in_contact = torch.sum(foot_contact.float(), dim=1)

        # 惩罚未着地的脚数量；六足都着地时为 0
        missing_contact_penalty = len(self.feet_indices) - num_feet_in_contact

        return missing_contact_penalty * small_command_mask.float()
    
    def _reward_lateral_lin_vel_y(self):
        """
        Penalize lateral (y-axis) velocity tracking error.
        Stronger penalty when forward command |cmd_x| is larger.
        """
        y_error = self.commands[:, 1] - self.base_lin_vel[:, 1]

        # 可在 cfg.rewards 中配置，没配就用默认值
        gain_with_cmd_x = getattr(self.cfg.rewards, "tracking_lin_vel_y_cmdx_gain", 0.25)

        # |cmd_x| 越大，惩罚越强（例如 cmd_x=4 时权重约 2x）
        dynamic_weight = 1.0 + gain_with_cmd_x * torch.abs(self.commands[:, 0])

        return dynamic_weight * torch.square(y_error)
