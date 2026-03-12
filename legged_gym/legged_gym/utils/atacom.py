"""
ATACOM 安全层实现（投影优化版）
对应文档章节：2.2.1 安全约束形式化 + 2.2.2 算法设计
运行时状态向量 s（S = 58 维）：
    [0 :18)   q          关节位置
    [18:36)   dq         关节速度
    [36:54)   tau        关节输出力矩
    [54:57)   phi        机身欧拉角（roll, pitch, yaw），ZYX 旋转顺序
    [57:58)   z_body     机身高度

约束向量 k（K = 77 维）：
    [0 :36)   关节位置限制（上下限各 18，交错排列）
    [36:54)   关节速度限制（18）
    [54:72)   关节力矩限制（18）
    [72:74)   机身高度限制（上限、下限各 1）
    [74:77)   机身倾角限制（三轴各 1）
"""

from typing import Tuple, Dict, Optional
import torch


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
S = 58    # 运行时状态维度
U = 18    # 控制输入维度（关节数）
K = 77    # 约束总数：36 + 18 + 18 + 2 + 3
U_EXT = U + K   # 增广控制维度 = 95


def _check_tensor(name: str, t: torch.Tensor, step: str = "") -> bool:
    """检查张量是否含 nan/inf（仅 debug 级别日志）。"""
    has_nan = torch.isnan(t).any()
    has_inf = torch.isinf(t).any()
    
    if has_nan or has_inf:
        valid = ~torch.isinf(t) & ~torch.isnan(t)
        prefix = f"[ATACOM{' ' + step if step else ''}] {name}"
        
        vmin = t[valid].min().item() if valid.any() else float('nan')
        vmax = t[valid].max().item() if valid.any() else float('nan')
        nan_count = torch.isnan(t).sum().item()
        inf_count = torch.isinf(t).sum().item()
        
        print(
            f"{prefix} 包含异常值! "
            f"shape={tuple(t.shape)} "
            f"有效范围=[{vmin:.4f}, {vmax:.4f}] "
            f"NaN={nan_count} Inf={inf_count}"
        )
        return True
    return False


class ATACOMSafetyLayer:
    """ATACOM 安全层（投影优化版）。

    调用 `forward(s, u_nominal)` 将 RL 策略输出的名义动作映射为安全动作。
    调用 `compute_info_scalars(info)` 将 forward 返回的 info 聚合为标量（触发 GPU 同步）。

    欧拉角约定：ZYX 旋转顺序（先绕 Z 转 yaw，再绕 Y 转 pitch，再绕 X 转 roll）。
    pitch 接近 ±90° 时映射矩阵奇异（万向锁），debug 模式下会打印警告。
    """

    def __init__(
        self,
        robot_params: Dict,
        lambda_retract: float = 1.0,
        beta: float = 1.0,
        dt: float = 0.01,
        debug_mode: bool = False,
        debug_level: str = 'basic',  # basic/verbose/debug
        debug_interval: int = 100,
    ):
        """
        Args:
            robot_params   : 机器人物理参数字典（见文件头说明）
            lambda_retract : 收缩增益 λ
            beta           : 松弛变量动力学系数
            dt             : 控制步长
            debug_mode     : 是否启用日志输出
            debug_level    : 日志级别:
                             - basic: 仅输出关键约束信息
                             - verbose: 输出详细约束违反信息
                             - debug: 输出所有张量检查和调试信息
            debug_interval : 日志输出间隔（步），-1 表示每步输出
        """
        self.robot_params = robot_params
        self.K = K
        self.U = U
        self.S = S
        self.lam = lambda_retract
        self.beta = beta
        self.dt = dt
        
        # 日志配置
        self.debug_mode = debug_mode
        self.debug_level = debug_level.lower()
        self.debug_interval = debug_interval
        self._step_count = 0

        # mu 上界：保证 beta * mu_max < 88，防止 exp 溢出（float32 约 e^88 ≈ 1.6e38）
        self.mu_max = 80.0 / max(self.beta, 1e-6)

        # 参考坐标系 T_ref 不再需要，但保留占位（实际未使用）
        self._T_ref_cpu = torch.zeros(U_EXT, U)
        self._T_ref_cpu[:U, :U] = torch.eye(U)

        # G 矩阵缓存（结构固定：G[18:36,:]=I, G[36:54,:]=I，其余为 0）
        self._G_cache: Optional[torch.Tensor] = None
        self._G_device: Optional[torch.device] = None

        # 预计算雅可比行列索引，延迟移到 device
        self._idx_device: Optional[torch.device] = None
        self._idx = {}

    # -----------------------------------------------------------------------
    # 内部工具：日志控制
    # -----------------------------------------------------------------------

    def _should_log(self, level: str = 'basic') -> bool:
        """判断当前是否应该输出日志。"""
        if not self.debug_mode:
            return False
            
        level_map = {'basic': 0, 'verbose': 1, 'debug': 2}
        current_level = level_map.get(self.debug_level, 0)
        required_level = level_map.get(level, 0)
        
        if current_level < required_level:
            return False
            
        if self.debug_interval <= 0:
            return True
        return (self._step_count % self.debug_interval) == 0

    # -----------------------------------------------------------------------
    # 设备感知的延迟初始化
    # -----------------------------------------------------------------------

    def _ensure_on_device(self, device: torch.device) -> None:
        """将缓存张量（G、索引）移到 device（仅首次或设备切换时执行）。"""
        if self._idx_device == device:
            return

        # ── G 矩阵缓存 ─────────────────────────────────────────────────────
        G = torch.zeros(1, S, U, device=device)
        eye_U = torch.eye(U, device=device)
        G[0, 18:36, :] = eye_U
        G[0, 36:54, :] = eye_U
        self._G_cache = G
        self._G_device = device

        # ── 雅可比预计算索引 ────────────────────────────────────────────────
        self._idx = {
            'pos_upper_rows': torch.arange(0,  36, 2, device=device),   # (18,)
            'pos_lower_rows': torch.arange(1,  36, 2, device=device),   # (18,)
            'joint_cols'    : torch.arange(0,  18,    device=device),   # (18,)
            'vel_rows'      : torch.arange(36, 54,    device=device),   # (18,)
            'vel_cols'      : torch.arange(18, 36,    device=device),   # (18,)
            'tau_rows'      : torch.arange(54, 72,    device=device),   # (18,)
            'tau_cols'      : torch.arange(36, 54,    device=device),   # (18,)
            'phi_rows'      : torch.arange(74, 77,    device=device),   # (3,)
            'phi_cols'      : torch.arange(54, 57,    device=device),   # (3,)
        }

        self._idx_device = device

    # -----------------------------------------------------------------------
    # 第一部分：约束函数 k(s)
    # -----------------------------------------------------------------------

    def compute_constraints(self, s: torch.Tensor) -> torch.Tensor:
        """
        计算约束向量 k(s)，k_i <= 0 表示第 i 个约束满足。

        Args:
            s: (num_envs, S=58)
        Returns:
            k: (num_envs, K=77)
        """
        assert s.shape[1] == S, f"状态维度应为 {S}，实际收到 {s.shape[1]}"

        rp = self.robot_params
        q_max = rp['q_max'].to(s.device)    # (18,)
        q_min = rp['q_min'].to(s.device)    # (18,)
        dq_max = rp['dq_max'].to(s.device)  # (18,)
        tau_max = rp['tau_max'].to(s.device) # (18,)
        phi_max = rp['phi_max'].to(s.device) # (3,)
        z_max = rp['z_max']                 # float
        z_min = rp['z_min']                 # float

        q = s[:, 0:18]    # (num_envs, 18)
        dq = s[:, 18:36]  # (num_envs, 18)
        tau = s[:, 36:54] # (num_envs, 18)
        phi = s[:, 54:57] # (num_envs, 3)
        z_body = s[:, 57] # (num_envs,)

        # (a) 关节位置限制 → 36 维，交错排列：[q0_upper, q0_lower, ...]
        k_pos = torch.stack([q - q_max, q_min - q], dim=2).reshape(s.shape[0], 36)

        # (b) 关节速度限制 → 18 维
        k_vel = torch.abs(dq) - dq_max

        # (c) 关节力矩限制 → 18 维
        k_tau = torch.abs(tau) - tau_max

        # (d) 机身高度限制 → 2 维
        k_z = torch.stack([z_body - z_max, z_min - z_body], dim=1)

        # (e) 机身倾角限制 → 3 维
        k_phi = torch.abs(phi) - phi_max

        k = torch.cat([k_pos, k_vel, k_tau, k_z, k_phi], dim=1)  # (num_envs, 77)
        assert k.shape[1] == K
        return k

    # -----------------------------------------------------------------------
    # 第二部分：约束雅可比 J_k = ∂k/∂s
    # -----------------------------------------------------------------------

    def compute_constraint_jacobian(self, s: torch.Tensor) -> torch.Tensor:
        """
        解析计算约束雅可比（全向量化版本）。

        Args:
            s: (num_envs, S=58)
        Returns:
            J_k: (num_envs, K=77, S=58)
        """
        num_envs = s.shape[0]
        device = s.device
        idx = self._idx   # 已在 _ensure_on_device 中移到 device

        J_k = torch.zeros((num_envs, K, S), device=device)

        # (a) 关节位置
        J_k[:, idx['pos_upper_rows'], idx['joint_cols']] =  1.0
        J_k[:, idx['pos_lower_rows'], idx['joint_cols']] = -1.0

        # (b) 关节速度：∂|dq_j|/∂dq_j = sign(dq_j)
        J_k[:, idx['vel_rows'], idx['vel_cols']] = torch.sign(s[:, idx['vel_cols']])

        # (c) 关节力矩：∂|tau_j|/∂tau_j = sign(tau_j)
        J_k[:, idx['tau_rows'], idx['tau_cols']] = torch.sign(s[:, idx['tau_cols']])

        # (d) 机身高度
        J_k[:, 72, 57] =  1.0
        J_k[:, 73, 57] = -1.0

        # (e) 机身倾角
        J_k[:, idx['phi_rows'], idx['phi_cols']] = torch.sign(s[:, idx['phi_cols']])

        return J_k  # (num_envs, 77, 58)

    # -----------------------------------------------------------------------
    # 第三部分：漂移项 f(s) 和输入增益 G(s)
    # -----------------------------------------------------------------------

    def compute_drift_f(
        self,
        s: torch.Tensor,
        ang_vel_body: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        漂移项 f(s)，(num_envs, S=58)。

        欧拉角速率映射（ZYX，roll=φ, pitch=θ）：
            φ̇   [1,  sin(φ)tan(θ),  cos(φ)tan(θ)] [ω_x]
            θ̇ = [0,  cos(φ),       -sin(φ)       ] [ω_y]
            ψ̇   [0,  sin(φ)/cos(θ), cos(φ)/cos(θ)] [ω_z]
        """
        num_envs = s.shape[0]
        f = torch.zeros((num_envs, S), device=s.device)

        # 位置导数 = 速度
        f[:, 0:18] = s[:, 18:36]

        if ang_vel_body is not None:
            euler = s[:, 54:57]
            roll = euler[:, 0]
            pitch = euler[:, 1]

            sr = torch.sin(roll)
            cr = torch.cos(roll)
            sp = torch.sin(pitch)
            cp = torch.cos(pitch)

            # 万向锁检测（仅 debug 级别输出）
            if self._should_log('debug'):
                gimbal_mask = cp.abs() < 0.1
                if gimbal_mask.any():
                    bad_pitch = pitch[gimbal_mask] * (180.0 / torch.pi)
                    print(
                        f"[ATACOM] 警告：{gimbal_mask.sum().item()} 个环境的 pitch "
                        f"接近万向锁奇点（|pitch| 最大 {bad_pitch.abs().max().item():.1f}°）"
                    )

            cp_safe = torch.where(cp.abs() < 1e-3, torch.full_like(cp, 1e-3), cp)
            inv_cp = 1.0 / cp_safe
            tan_p = sp * inv_cp

            zeros = torch.zeros_like(sr)
            ones = torch.ones_like(sr)
            T_row0 = torch.stack([ones,  sr * tan_p,  cr * tan_p ], dim=1)
            T_row1 = torch.stack([zeros, cr,          -sr         ], dim=1)
            T_row2 = torch.stack([zeros, sr * inv_cp,  cr * inv_cp], dim=1)
            T = torch.stack([T_row0, T_row1, T_row2], dim=1)   # (N, 3, 3)

            f[:, 54:57] = torch.bmm(T, ang_vel_body.unsqueeze(-1)).squeeze(-1)

        return f

    def compute_input_gain_G(self, s: torch.Tensor) -> torch.Tensor:
        """
        输入增益矩阵 G(s)，(num_envs, S=58, U=18)。

        G[18:36, :] = I_18，G[36:54, :] = I_18，其余为 0。
        结构固定，返回预分配缓存的 expand 视图（零拷贝）。
        """
        return self._G_cache.expand(s.shape[0], -1, -1)   # 零拷贝广播

    # -----------------------------------------------------------------------
    # 第四部分：松弛变量动力学 A(μ)
    # -----------------------------------------------------------------------

    def compute_slack_dynamics_A(self, mu: torch.Tensor) -> torch.Tensor:
        """
        松弛变量动力学对角矩阵 A(μ)，(num_envs, K, K)。

        alpha_i(mu_i) = exp(beta * mu_i) - 1，clamp 到 [1e-3, 1e6]。
        """
        alpha = torch.exp(self.beta * mu) - 1.0
        alpha = torch.clamp(alpha, min=1e-3, max=1e6)
        return torch.diag_embed(alpha)   # (num_envs, K, K)

    # -----------------------------------------------------------------------
    # 内部工具：鲁棒伪逆（Cholesky 优先，批量 fallback）
    # -----------------------------------------------------------------------

    @staticmethod
    def _robust_pinv(J: torch.Tensor, rcond: float = 1e-4) -> torch.Tensor:
        """
        批量鲁棒伪逆（v2：Cholesky 优先）。

        路径优先级：
          1. Cholesky 右逆：J^+ = J^T (J J^T + eps·I)^{-1}
               用 torch.linalg.cholesky_solve，比 SVD 快约 2-3×
               适用条件：J J^T 正定（通常成立，eps 保证数值正定）
          2. 批量 pinv（SVD）：Cholesky 失败时 fallback
          3. 批量 Tikhonov inv：进一步 fallback
          4. 批量 lstsq：最终 fallback

        Args:
            J     : (num_envs, K=77, U_ext=95)
        Returns:
            J_pinv: (num_envs, U_ext=95, K=77)
        """
        num_envs, K_dim, _ = J.shape
        device = J.device
        dtype = J.dtype

        # ── 路径 1：Cholesky 右逆（最快）───────────────────────────────────
        JJT = torch.bmm(J, J.mT)   # (num_envs, K, K)

        # 自适应正则化：eps = rcond * max(1, ||JJT||_F / K)
        frob = torch.linalg.matrix_norm(JJT)              # (num_envs,)
        eps = rcond * torch.clamp(frob / K_dim, min=1.0)  # (num_envs,)
        reg = eps.view(num_envs, 1, 1) * torch.eye(
            K_dim, device=device, dtype=dtype
        ).unsqueeze(0)                                     # (num_envs, K, K)

        JJT_reg = JJT + reg   # (num_envs, K, K)，正定

        try:
            # cholesky_solve: 解 (JJT_reg) X = I，即 X = JJT_reg^{-1}
            L = torch.linalg.cholesky(JJT_reg)   # (num_envs, K, K)
            Id = torch.eye(K_dim, device=device, dtype=dtype).unsqueeze(0).expand(num_envs, -1, -1)
            JJT_inv = torch.cholesky_solve(Id, L)   # (num_envs, K, K)
            return torch.bmm(J.mT, JJT_inv)          # (num_envs, U_ext, K)
        except Exception:
            pass   # 静默 fallback，不打印（避免每步输出）

        # ── 路径 2：批量 SVD pinv（通用）───────────────────────────────────
        try:
            return torch.linalg.pinv(J, rcond=rcond)
        except Exception as e:
            print(f"[ATACOM] 批量 pinv 失败（{e}），切换 Tikhonov 模式")

        # ── 路径 3：批量 Tikhonov inv ───────────────────────────────────────
        try:
            JJT_inv = torch.linalg.inv(JJT_reg)
            return torch.bmm(J.mT, JJT_inv)
        except Exception as e:
            print(f"[ATACOM] 批量 Tikhonov inv 失败（{e}），切换 lstsq 模式")

        # ── 路径 4：批量 lstsq（最终 fallback）─────────────────────────────
        Id = torch.eye(K_dim, device=device, dtype=dtype).unsqueeze(0).expand(num_envs, -1, -1)
        return torch.linalg.lstsq(J, Id).solution

    # -----------------------------------------------------------------------
    # 内部工具：日志输出约束违反详情
    # -----------------------------------------------------------------------

    def _log_constraint_violations(self, k: torch.Tensor, num_envs: int) -> None:
        """输出约束违反详情（仅 verbose/debug 级别）。"""
        if not self._should_log('verbose'):
            return
            
        violated = k > 0
        if not violated.any():
            return

        rp = self.robot_params
        constraint_names = []
        
        # 构建约束名称列表
        q_max = rp['q_max']
        q_min = rp['q_min']
        dq_max = rp['dq_max']
        tau_max = rp['tau_max']
        phi_max = rp['phi_max']

        for j in range(18):
            constraint_names.append(f"q[{j}]_upper  (上限={q_max[j].item():.3f})")
            constraint_names.append(f"q[{j}]_lower  (下限={q_min[j].item():.3f})")
        for j in range(18):
            constraint_names.append(f"dq[{j}]_abs   (上限=±{dq_max[j].item():.3f})")
        for j in range(18):
            constraint_names.append(f"tau[{j}]_abs  (上限=±{tau_max[j].item():.3f})")
        constraint_names.append(f"z_upper (上限={rp['z_max']:.3f})")
        constraint_names.append(f"z_lower (下限={rp['z_min']:.3f})")
        for i, axis in enumerate(['roll', 'pitch', 'yaw']):
            constraint_names.append(f"phi_{axis}_abs (上限=±{phi_max[i].item():.3f})")

        # 计算每个约束的最大违反值
        k_max_per = k.max(dim=0).values
        
        print("\n[ATACOM] 约束违反详情：")
        violation_count = 0
        for idx, (name, val) in enumerate(zip(constraint_names, k_max_per)):
            if val.item() > 0:
                n_viol = violated[:, idx].sum().item()
                violation_count += 1
                print(
                    f"  [{idx:02d}] {name:<35s}  "
                    f"最大违反量={val.item():.4e}  "
                    f"违反环境数={int(n_viol)}/{num_envs}"
                )
        
        if violation_count == 0:
            print("  无约束违反")
        print("-" * 80)

    # -----------------------------------------------------------------------
    # 第六部分：ATACOM 前向计算（投影优化版）
    # -----------------------------------------------------------------------

    def forward(
        self,
        s: torch.Tensor,
        u_nominal: torch.Tensor,
        ang_vel_body: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        将名义动作 u_nominal 映射为满足约束的安全动作 u_safe。

        Args:
            s            : 当前状态 (num_envs, S=58)
            u_nominal    : 名义动作 (num_envs, U=18)
            ang_vel_body : 机体系角速度 [ω_x, ω_y, ω_z]（rad/s），(num_envs, 3)。

        Returns:
            u_safe : 安全动作 (num_envs, U=18)
            u_mu   : 松弛变量更新量 (num_envs, K=77)
            info   : 持有 tensor 引用的字典，不含 .item() 调用。
                     调用 compute_info_scalars(info) 可聚合为标量。
        """
        device = s.device
        num_envs = s.shape[0]
        
        # 延迟初始化缓存（首次或 device 切换时执行）
        self._ensure_on_device(device)

        # ── 输入张量检查（仅 debug 级别）────────────────────────────────────
        if self._should_log('debug'):
            _check_tensor("输入状态 s", s, "Step0")
            _check_tensor("名义动作 u_nominal", u_nominal, "Step0")
            if ang_vel_body is not None:
                _check_tensor("机体系角速度 ang_vel_body", ang_vel_body, "Step0")

        # Step 1: 约束值 k(s)
        k = self.compute_constraints(s)  # (N, 77)

        # ── 输出约束概览信息 ───────────────────────────────────────────────
        if self._should_log('basic'):
            k_min = k.min().item()
            k_max = k.max().item()
            k_viol = torch.clamp(k, min=0)
            total_violation = k_viol.sum(dim=1).mean().item()
            
            print(
                f"\n[ATACOM Step {self._step_count}] "
                f"约束概览 - 范围: [{k_min:.3e}, {k_max:.3e}] | "
                f"平均违反量: {total_violation:.3e}"
            )
            
            # 输出详细的约束违反信息（verbose/debug 级别）
            self._log_constraint_violations(k, num_envs)

        # Step 2: 松弛变量 μ
        mu = torch.clamp(-k, min=1e-6, max=self.mu_max)  # (N, 77)

        # Step 3: 等式约束残差 c = k + μ
        c = k + mu  # (N, 77)

        # Step 4: 约束雅可比
        J_k = self.compute_constraint_jacobian(s)  # (N, 77, 58)

        # Step 5: 漂移项与输入增益
        f = self.compute_drift_f(s, ang_vel_body=ang_vel_body)  # (N, 58)
        G = self.compute_input_gain_G(s)  # (N, 58, 18)，零拷贝

        # Debug 级别检查漂移项
        if self._should_log('debug'):
            _check_tensor("漂移项 f", f, "Step5")

        # Step 6: 约束漂移 ψ = clip(J_k @ f, 0)
        psi = torch.clamp(
            torch.bmm(J_k, f.unsqueeze(-1)).squeeze(-1), min=0.0
        )  # (N, 77)

        # Step 7: 松弛变量动力学矩阵 A(μ)
        A = self.compute_slack_dynamics_A(mu)  # (N, 77, 77)

        # Step 8: 输入雅可比 J_u = [J_k @ G, A]
        J_G = torch.bmm(J_k, G)  # (N, 77, 18)
        J_u = torch.cat([J_G, A], dim=2)  # (N, 77, 95)

        # Debug 级别检查输入雅可比
        if self._should_log('debug'):
            _check_tensor("输入雅可比 J_u", J_u, "Step8")

        # Step 9: 伪逆（Cholesky 优先）
        J_u_pinv = self._robust_pinv(J_u)  # (N, 95, 77)

        # Debug 级别检查伪逆
        if self._should_log('debug'):
            if _check_tensor("伪逆 J_u_pinv", J_u_pinv, "Step9"):
                cond = torch.linalg.cond(J_u)
                print(
                    f"[ATACOM] J_u 条件数: "
                    f"最小值={cond.min().item():.2e} 最大值={cond.max().item():.2e}"
                )
                J_u_pinv = torch.nan_to_num(J_u_pinv, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 10: 增广名义动作 [u_nominal; 0_K]
        zeros_K = torch.zeros((num_envs, K), device=device)
        u_nom_ext = torch.cat([u_nominal, zeros_K], dim=1)  # (N, 95)

        # Step 11: 计算 J_u @ u_nom_ext（用于投影）
        J_u_u_nom = torch.bmm(J_u, u_nom_ext.unsqueeze(-1)).squeeze(-1)  # (N, 77)

        # Step 12: 合并右端项：-(ψ + λ c) + J_u_u_nom
        rhs = -(psi + self.lam * c)  # (N, 77)
        combined_rhs = (rhs + J_u_u_nom).unsqueeze(-1)  # (N, 77, 1)

        # Step 13: 计算增广安全动作（投影公式）
        u_ext = u_nom_ext - torch.bmm(J_u_pinv, combined_rhs).squeeze(-1)  # (N, 95)

        # Step 14: 分离关节动作与松弛变量
        u_safe = u_ext[:, :U]  # (N, 18)
        u_mu = u_ext[:, U:]    # (N, 77)

        # ── 输出安全检查 ───────────────────────────────────────────────────
        # 批量检测 nan/inf，减少 GPU 同步
        u_safe_valid = torch.isfinite(u_safe).all()
        u_mu_valid = torch.isfinite(u_mu).all()

        if not u_safe_valid:
            if self._should_log('verbose'):
                print("[ATACOM] u_safe 包含 nan/inf，已重置为零动作")
            u_safe = torch.zeros_like(u_safe)

        if not u_mu_valid:
            if self._should_log('verbose'):
                print("[ATACOM] u_mu 包含 nan/inf，已重置为零")
            u_mu = torch.zeros_like(u_mu)

        # ── info 字典：持有 tensor 引用，不调用 .item() ────────────────────
        info = {
            'k': k,          # (N, 77) 约束值
            'mu': mu,        # (N, 77) 松弛变量
            'psi': psi,      # (N, 77) 约束漂移
            'u_mu': u_mu,    # (N, 77) 松弛变量更新量
            'step': self._step_count,
        }

        self._step_count += 1
        return u_safe, u_mu, info

    # -----------------------------------------------------------------------
    # 公共工具：按需聚合 info 标量（触发 GPU 同步，建议每 N 步调用一次）
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_info_scalars(info: Dict) -> Dict:
        """
        将 forward() 返回的 info 聚合为 Python 标量。
        """
        k = info['k']
        mu = info['mu']
        psi = info['psi']
        u_mu = info['u_mu']

        k_viol = torch.clamp(k, min=0)
        return {
            'constraint_violation': k_viol.sum(dim=1).mean().item(),
            'violation_pos': k_viol[:, 0:36].sum(dim=1).mean().item(),
            'violation_vel': k_viol[:, 36:54].sum(dim=1).mean().item(),
            'violation_tau': k_viol[:, 54:72].sum(dim=1).mean().item(),
            'violation_height': k_viol[:, 72:74].sum(dim=1).mean().item(),
            'violation_tilt': k_viol[:, 74:77].sum(dim=1).mean().item(),
            'mu_norm': mu.norm(dim=1).mean().item(),
            'u_mu_norm': u_mu.norm(dim=1).mean().item(),
            'u_mu_pos': u_mu[:, 0:36].norm(dim=1).mean().item(),
            'u_mu_vel': u_mu[:, 36:54].norm(dim=1).mean().item(),
            'u_mu_tau': u_mu[:, 54:72].norm(dim=1).mean().item(),
            'u_mu_height': u_mu[:, 72:74].norm(dim=1).mean().item(),
            'u_mu_tilt': u_mu[:, 74:77].norm(dim=1).mean().item(),
            'safe_ratio': (k <= 0).all(dim=1).float().mean().item(),
            'psi_norm': psi.norm(dim=1).mean().item(),
            'step': info['step'],
        }