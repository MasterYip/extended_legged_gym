from typing import Dict, Tuple

import torch

from legged_gym.envs.el_4090.el_4090 import EL_4090
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
    """

    # 父类原始观测维度，用于 obs_buf 裁剪保护
    _BASE_OBS_DIM = 66

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless,
                 task_name="el4090_spider"):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,
                         task_name=task_name)

        safety_cfg = getattr(self.cfg, 'safety', El4090SafeCfg.safety)

        def _to_tensor_18(x, default):
            """将标量 / list / tensor 统一转为 (18,) float32 tensor。"""
            val = getattr(safety_cfg, x, default)
            if isinstance(val, (int, float)):
                val = [val] * 18
            return torch.tensor(val, device=self.device, dtype=torch.float32)

        def _to_tensor_3(x, default):
            """将标量 / list / tensor 统一转为 (3,) float32 tensor。"""
            val = getattr(safety_cfg, x, default)
            if isinstance(val, (int, float)):
                val = [val] * 3
            return torch.tensor(val, device=self.device, dtype=torch.float32)

        robot_params: Dict = {
            'q_max'  : _to_tensor_18('q_max',   [3.14]  * 18),
            'q_min'  : _to_tensor_18('q_min',   [-3.14] * 18),
            'dq_max' : _to_tensor_18('dq_max',  [10.0]  * 18),
            'tau_max': _to_tensor_18('tau_max',  [100.0] * 18),
            'phi_max': _to_tensor_3( 'phi_max',  [0.26]  * 3),
            'z_max'  : float(getattr(safety_cfg, 'z_max',  1.0)),
            'z_min'  : float(getattr(safety_cfg, 'z_min',  0.1)),
        }

        self.atacom = ATACOMSafetyLayer(
            robot_params=robot_params,
            lambda_retract=float(getattr(safety_cfg, 'lambda_retract', 1.0)),
            beta=float(getattr(safety_cfg, 'beta', 1.0)),
            dt=float(getattr(safety_cfg, 'dt', self.dt)),
        )

        self._atacom_enabled      = bool(getattr(safety_cfg, 'enable_atacom',        True))
        self._atacom_clip_nominal = bool(getattr(safety_cfg, 'clip_nominal_actions',  True))
        self._atacom_warmup       = int(getattr(safety_cfg,  'warmup_steps',          0))
        self._atacom_log_info     = bool(getattr(safety_cfg, 'log_info',              True))

    # ------------------------------------------------------------------
    # 状态拼装
    # ------------------------------------------------------------------

    def _build_atacom_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """从仿真 buffer 拼装 ATACOM 运行时状态 s，(num_envs, 58)。

        数据来源（legged_gym 标准 buffer）：
            dof_pos        self.dof_pos          (num_envs, num_dof)
            dof_vel        self.dof_vel          (num_envs, num_dof)
            torques        self.torques          (num_envs, num_dof)  上一步实际力矩
            base_quat      root_states[:, 3:7]   机身四元数
            ang_vel_world  root_states[:, 10:13] 世界系角速度
            base_pos_z     root_states[:, 2]     机身高度

        Returns:
            s            : ATACOM 状态向量 (num_envs, 58)
            ang_vel_base : 机体系角速度 (num_envs, 3)，供 ATACOM 漂移项使用，
                           避免在 step 中重复计算
        """
        num_envs = self.num_envs
        device   = self.device
        s = torch.zeros((num_envs, 58), device=device)

        # [0:18)  关节位置
        s[:, 0:18] = self.dof_pos[:, :18]

        # [18:36) 关节速度
        s[:, 18:36] = self.dof_vel[:, :18]

        # [36:54) 关节实际力矩
        # self.torques 是 _compute_torques 的输出，即上一控制步实际施加的力矩
        # 第一步时 torques 可能尚未初始化，用零填充
        if hasattr(self, 'torques') and self.torques is not None:
            s[:, 36:54] = self.torques[:, :18]
        # else: 保持零（已由 torch.zeros 初始化）

        # [54:57) 机体系欧拉角（roll, pitch, yaw），ZYX 旋转顺序
        # 同时计算机体系角速度，供调用方传给 ATACOM 漂移项，避免重复计算
        base_quat     = self.root_states[:, 3:7]    # (num_envs, 4)  (x, y, z, w)
        ang_vel_world = self.root_states[:, 10:13]  # (num_envs, 3)
        ang_vel_base  = quat_rotate_inverse(base_quat, ang_vel_world)  # (num_envs, 3)

        # 四元数 (x, y, z, w) -> 欧拉角 (roll, pitch, yaw)，ZYX 顺序
        qx = base_quat[:, 0]
        qy = base_quat[:, 1]
        qz = base_quat[:, 2]
        qw = base_quat[:, 3]

        # roll (绕 X 轴)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # pitch (绕 Y 轴)
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)

        # yaw (绕 Z 轴)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        s[:, 54:57] = torch.stack([roll, pitch, yaw], dim=1)

        # [57:58) 机身高度
        s[:, 57] = self.root_states[:, 2]

        # 修正1：返回 ang_vel_base，避免 step 中重复计算
        return s, ang_vel_base

    # ------------------------------------------------------------------
    # buffer 初始化
    # ------------------------------------------------------------------

    def _init_buffers(self):
        """初始化父类 buffer，并额外分配 u_mu buffer。"""
        super()._init_buffers()
        # u_mu：松弛变量更新量，(num_envs, K=77)
        # 第一步 ATACOM 尚未运行时保持全零，不影响观测合法性
        self.u_mu = torch.zeros(
            (self.num_envs, 77), device=self.device, dtype=torch.float32
        )

    # ------------------------------------------------------------------
    # 观测计算
    # ------------------------------------------------------------------

    def _get_noise_scale_vec(self, cfg):
        """覆写父类，强制按父类原始 66 维构造噪声向量。

        父类 _init_buffers 调用本函数时 obs_buf 已是 143 维（num_observations=143），
        若不覆写，noise_scale_vec 会被构造成 143 维，之后 compute_observations
        把 obs_buf 裁回 66 维再加噪时就会 66 vs 143 报错。
        这里临时将 obs_buf 替换为 66 维的零张量，让父类按 66 维构造噪声向量，
        构造完毕后再还原。
        """
        original_obs_buf = self.obs_buf
        self.obs_buf = torch.zeros(
            (self.num_envs, self._BASE_OBS_DIM),
            device=self.device, dtype=torch.float32
        )
        noise_vec = super()._get_noise_scale_vec(cfg)   # (66,)
        self.obs_buf = original_obs_buf
        return noise_vec  # 始终是 66 维，与父类加噪时的 obs_buf 匹配

    def compute_observations(self):
        """在父类观测基础上追加 u_mu（77维松弛变量更新量）。

        观测结构（共 143 维，与 config num_observations=143 对应）：
            [0  :3 )  base_lin_vel        3
            [3  :6 )  base_ang_vel        3
            [6  :9 )  projected_gravity   3
            [9  :12)  commands            3
            [12 :30)  dof_pos             18
            [30 :48)  dof_vel             18
            [48 :66)  actions             18
            [66 :143) u_mu                77   ← 新增

        执行顺序：
            1. 将 obs_buf 裁回 66 维，防止父类加噪时与 noise_scale_vec(66) 不匹配
            2. super() 填充 obs_buf(66维) 并原地加噪
            3. cat u_mu → obs_buf 扩展为 143 维
        """
        # 父类 compute_observations 用 += 原地加噪，
        # 若 obs_buf 仍是上一帧的 143 维会与 noise_scale_vec(66) 维度不匹配报错。
        # 提前裁回 66 维保证父类加噪正常工作。
        if self.obs_buf.shape[1] != self._BASE_OBS_DIM:
            self.obs_buf = self.obs_buf[:, :self._BASE_OBS_DIM].contiguous()

        super().compute_observations()              # obs_buf: (num_envs, 66)，已加噪
        self.obs_buf = torch.cat([self.obs_buf, self.u_mu], dim=-1)  # (num_envs, 143)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, actions, *args, **kwargs) -> Tuple:
        """ATACOM 安全过滤后转发给父类 step。"""

        # 未启用或处于 warmup 阶段，直接走原始逻辑
        if (not self._atacom_enabled
                or self.common_step_counter < self._atacom_warmup):
            return super().step(actions, *args, **kwargs)

        # 转 tensor
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        else:
            actions = actions.to(self.device)

        # 可选：对名义动作做范围裁剪（与训练时保持一致）
        if self._atacom_clip_nominal:
            clip_val = getattr(self.cfg.normalization, 'clip_actions', None)
            if clip_val is not None:
                actions = torch.clamp(actions, -clip_val, clip_val)

        # 保证形状为 (num_envs, 18)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0).expand(self.num_envs, -1)

        # 修正1：直接复用 _build_atacom_state 返回的 ang_vel_base，不再重复计算
        s, ang_vel_base = self._build_atacom_state()
        u_safe, u_mu, atacom_info = self.atacom.forward(s, actions, ang_vel_body=ang_vel_base)

        # 保存 u_mu 供本步 compute_observations 使用
        self.u_mu = u_mu

        # 将约束监控信息写入 extras 供 tensorboard 记录
        # log_info=True（默认）：写入所有标量指标，用于训练监控
        # log_info=False：跳过写入，减少 runner 的 logging 开销
        if self._atacom_log_info:
            if not hasattr(self, 'extras'):
                self.extras = {}
            # 只写入标量，跳过张量（constraint_value / u_mu 张量不适合 tensorboard）
            self.extras['atacom'] = {
                k: v for k, v in atacom_info.items()
                if not isinstance(v, torch.Tensor)
            }

        # 修正2：删除调试用 print("11111111")

        return super().step(u_safe, *args, **kwargs)