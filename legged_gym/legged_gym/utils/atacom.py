"""
ATACOM 安全层实现
对应文档章节：2.2.1 安全约束形式化 + 2.2.2 算法设计

运行时状态向量 s（S = 58 维）：
    [0 :18)   q          关节位置
    [18:36)   dq         关节速度
    [36:54)   tau        关节输出力矩
    [54:57)   phi        机身欧拉角（roll, pitch, yaw），ZYX 旋转顺序
    [57:58)   z_body     机身高度

所有限位参数保存在 robot_params 中，不占 s 的列：
    q_max   (18,)   关节位置上限
    q_min   (18,)   关节位置下限
    dq_max  (18,)   关节速度上限
    tau_max (18,)   关节力矩上限
    phi_max  (3,)   机身三轴倾角上限（rad）
    z_max   float   机身高度上限
    z_min   float   机身高度下限

约束向量 k（K = 77 维）：
    [0 :36)   关节位置限制（上下限各 18）
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
S = 58   # 运行时状态维度
U = 18   # 控制输入维度（关节数）
K = 77   # 约束总数：36 + 18 + 18 + 2 + 3


def _check(name: str, t: torch.Tensor, step: str = "") -> bool:
    """检查张量是否含 nan/inf，有则打印详细信息并返回 True。"""
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    if has_nan or has_inf:
        valid = ~torch.isinf(t) & ~torch.isnan(t)
        prefix = f"[ATACOM{' ' + step if step else ''}] {name}"
        vmin = t[valid].min().item() if valid.any() else float('nan')
        vmax = t[valid].max().item() if valid.any() else float('nan')
        print(
            f"{prefix} 含 {'nan ' if has_nan else ''}{'inf' if has_inf else ''}!"
            f"  shape={tuple(t.shape)}"
            f"  valid_min={vmin:.4f}  valid_max={vmax:.4f}"
            f"  nan_count={torch.isnan(t).sum().item()}"
            f"  inf_count={torch.isinf(t).sum().item()}"
        )
        return True
    return False


class ATACOMSafetyLayer:
    """ATACOM 安全层。

    调用 `forward(s, u_nominal)` 将 RL 策略输出的名义动作映射为安全动作。

    欧拉角约定：ZYX 旋转顺序（即先绕 Z 转 yaw，再绕 Y 转 pitch，再绕 X 转 roll），
    与大多数机器人框架（ROS、Isaac Gym 等）保持一致。
    机体角速度到欧拉角速率的映射矩阵 T(φ) 在 pitch 接近 ±90° 时奇异（万向锁），
    此时会打印警告；长期建议改用四元数表示姿态以彻底规避该问题。
    """

    def __init__(
        self,
        robot_params: Dict,
        lambda_retract: float = 1.0,
        beta: float = 1.0,
        dt: float = 0.01,
    ):
        """
        Args:
            robot_params   : 机器人物理参数字典（见文件头说明）
            lambda_retract : 收缩增益 λ
            beta           : 松弛变量动力学系数
            dt             : 控制步长
        """
        self.robot_params = robot_params
        self.K = K
        self.U = U
        self.S = S
        self.lam = lambda_retract
        self.beta = beta
        self.dt = dt

        # mu 上界：保证 beta * mu_max < 88，防止 exp 溢出（float32 约 e^88 ≈ 1.6e38）
        self.mu_max = 80.0 / max(self.beta, 1e-6)

        U_ext = U + K  # 增广控制维度 = 18 + 77 = 95
        # 参考坐标系 T_ref (U_ext, U)：前 U 行为单位阵，其余为 0
        self.T_ref = torch.zeros(U_ext, U)
        self.T_ref[:U, :U] = torch.eye(U)

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
        # 修正1：恢复输入维度断言，状态结构变更时能及早发现上游传参错误
        assert s.shape[1] == S, f"状态维度应为 {S}，实际收到 {s.shape[1]}"

        rp = self.robot_params
        q_max   = rp['q_max'].to(s.device)    # (18,)
        q_min   = rp['q_min'].to(s.device)    # (18,)
        dq_max  = rp['dq_max'].to(s.device)   # (18,)
        tau_max = rp['tau_max'].to(s.device)   # (18,)
        phi_max = rp['phi_max'].to(s.device)   # (3,)
        z_max   = rp['z_max']                  # float
        z_min   = rp['z_min']                  # float

        q      = s[:, 0:18]    # (num_envs, 18)
        dq     = s[:, 18:36]   # (num_envs, 18)
        tau    = s[:, 36:54]   # (num_envs, 18)
        phi    = s[:, 54:57]   # (num_envs, 3)  机体系欧拉角 (roll, pitch, yaw)，ZYX 顺序
        z_body = s[:, 57]      # (num_envs,)

        k_list = []

        # (a) 关节位置限制：上限 q_j - q_max_j，下限 q_min_j - q_j
        for j in range(18):
            k_list.append(q[:, j] - q_max[j])
            k_list.append(q_min[j] - q[:, j])

        # (b) 关节速度限制：|dq_j| - dq_max_j
        for j in range(18):
            k_list.append(torch.abs(dq[:, j]) - dq_max[j])

        # (c) 关节力矩限制：|tau_j| - tau_max_j
        for j in range(18):
            k_list.append(torch.abs(tau[:, j]) - tau_max[j])

        # (d) 机身高度限制：上限 z - z_max，下限 z_min - z
        k_list.append(z_body - z_max)
        k_list.append(z_min - z_body)

        # (e) 机身倾角限制：|phi_i| - phi_max_i（直接约束欧拉角角度）
        for i in range(3):
            k_list.append(torch.abs(phi[:, i]) - phi_max[i])

        k = torch.stack(k_list, dim=1)  # (num_envs, 77)
        assert k.shape[1] == K, f"约束维度不匹配：期望 {K}，实际 {k.shape[1]}"
        return k

    # -----------------------------------------------------------------------
    # 第二部分：约束雅可比 J_k = ∂k/∂s
    # -----------------------------------------------------------------------

    def compute_constraint_jacobian(self, s: torch.Tensor) -> torch.Tensor:
        """
        解析计算约束雅可比。

        Args:
            s: (num_envs, S=58)
        Returns:
            J_k: (num_envs, K=77, S=58)
        """
        num_envs = s.shape[0]
        J_k = torch.zeros((num_envs, K, S), device=s.device)

        idx = 0

        # (a) 关节位置限制
        for j in range(18):
            J_k[:, idx, j] = 1.0    # ∂(q_j - q_max)/∂q_j = +1
            idx += 1
            J_k[:, idx, j] = -1.0   # ∂(q_min - q_j)/∂q_j = -1
            idx += 1

        # (b) 关节速度限制：∂|dq_j|/∂dq_j = sign(dq_j)
        for j in range(18):
            J_k[:, idx, 18 + j] = torch.sign(s[:, 18 + j])
            idx += 1

        # (c) 关节力矩限制：∂|tau_j|/∂tau_j = sign(tau_j)
        for j in range(18):
            J_k[:, idx, 36 + j] = torch.sign(s[:, 36 + j])
            idx += 1

        # (d) 机身高度限制：z_body 位于状态索引 57
        J_k[:, idx, 57] = 1.0    # ∂(z - z_max)/∂z = +1
        idx += 1
        J_k[:, idx, 57] = -1.0   # ∂(z_min - z)/∂z = -1
        idx += 1

        # (e) 机身倾角限制：phi 位于状态索引 54~56，ZYX 顺序
        for i in range(3):
            J_k[:, idx, 54 + i] = torch.sign(s[:, 54 + i])
            idx += 1

        assert idx == K, f"雅可比行计数不匹配：{idx} != {K}"
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

        状态各通道的自然导数：
            f[0:18]  = dq         （位置导数 = 速度）
            f[18:36] = 0          （非线性项简化，待替换为真实动力学）
            f[36:54] = 0          （力矩由策略直接指定）
            f[54:57] = T(φ)@ω    （机体角速度→欧拉角速率，ZYX 顺序）
            f[57:58] = 0          （高度动力学简化）

        欧拉角速率映射（ZYX，roll=φ, pitch=θ）：
            φ̇   [1,  sin(φ)tan(θ),  cos(φ)tan(θ)] [ω_x]
            θ̇ = [0,  cos(φ),       -sin(φ)       ] [ω_y]
            ψ̇   [0,  sin(φ)/cos(θ), cos(φ)/cos(θ)] [ω_z]

        Args:
            s           : 当前状态 (num_envs, S=58)
            ang_vel_body: 机体系角速度 [ω_x, ω_y, ω_z]，(num_envs, 3)。
                          为 None 时 f[54:57] 保持 0（简化模式）。

        Notes:
            pitch 接近 ±90° 时映射矩阵奇异（万向锁），会打印警告。
            长期建议改用四元数表示姿态以彻底规避该问题。
        """
        num_envs = s.shape[0]
        f = torch.zeros((num_envs, S), device=s.device)

        # 位置导数 = 速度
        f[:, 0:18] = s[:, 18:36]

        # 速度导数的非线性项（待替换为真实动力学）：
        # tau_nonlinear = compute_gravity_torque(...) + compute_coriolis_torque(...)
        # M_inv = compute_inertia_matrix_inv(...)
        # f[:, 18:36] = torch.bmm(M_inv, tau_nonlinear.unsqueeze(-1)).squeeze(-1)

        if ang_vel_body is not None:
            # s[:, 54:57] 存储 [roll(φ), pitch(θ), yaw(ψ)]，ZYX 旋转顺序
            euler = s[:, 54:57]
            roll  = euler[:, 0]   # φ
            pitch = euler[:, 1]   # θ

            sr = torch.sin(roll)
            cr = torch.cos(roll)
            sp = torch.sin(pitch)
            cp = torch.cos(pitch)

            # 修正2：万向锁奇点警告（pitch 接近 ±90° 时 cos(pitch)→0，T(φ) 奇异）
            gimbal_mask = cp.abs() < 0.1  # 约 ±84° 以内开始退化
            if gimbal_mask.any():
                bad_count = gimbal_mask.sum().item()
                bad_pitch = pitch[gimbal_mask] * (180.0 / torch.pi)
                print(
                    f"[ATACOM compute_drift_f] 警告：{bad_count} 个环境的 pitch 接近万向锁奇点"
                    f"（|pitch| 最大 {bad_pitch.abs().max().item():.1f}°），"
                    f"欧拉角速率映射结果不可靠。建议改用四元数表示姿态。"
                )

            # 修正3：clamp 防止除零，但奇点附近结果已由上方警告标记为不可靠
            cp_safe = torch.clamp(cp.abs(), min=1e-3) * torch.sign(cp)
            # sign(cp) 保留符号，防止 pitch 在 (-90°,0) 时 clamp 改变符号导致映射方向错误
            # 当 cp 极小且为负时直接取 -1e-3，保证 inv_cp 不爆炸且符号正确
            cp_safe = torch.where(cp.abs() < 1e-3, torch.full_like(cp, 1e-3), cp)
            inv_cp  = 1.0 / cp_safe
            tan_p   = sp * inv_cp  # tan(pitch)

            # 修正4：按 ZYX 顺序（与文件头、类 docstring 约定一致）构造映射矩阵 T(φ)
            #
            #   φ̇   [1,  sin(φ)·tan(θ),  cos(φ)·tan(θ)] [ω_x]
            #   θ̇ = [0,  cos(φ),         -sin(φ)        ] [ω_y]
            #   ψ̇   [0,  sin(φ)/cos(θ),  cos(φ)/cos(θ) ] [ω_z]
            #
            T = torch.zeros((num_envs, 3, 3), device=s.device)

            T[:, 0, 0] = 1.0
            T[:, 0, 1] = sr * tan_p   # sin(φ)·tan(θ)
            T[:, 0, 2] = cr * tan_p   # cos(φ)·tan(θ)

            T[:, 1, 0] = 0.0
            T[:, 1, 1] = cr            # cos(φ)
            T[:, 1, 2] = -sr           # -sin(φ)

            T[:, 2, 0] = 0.0
            T[:, 2, 1] = sr * inv_cp   # sin(φ)/cos(θ)
            T[:, 2, 2] = cr * inv_cp   # cos(φ)/cos(θ)

            phi_dot = torch.bmm(T, ang_vel_body.unsqueeze(-1)).squeeze(-1)
            f[:, 54:57] = phi_dot

        return f

    def compute_input_gain_G(self, s: torch.Tensor) -> torch.Tensor:
        """
        输入增益矩阵 G(s)，(num_envs, S=58, U=18)。

        控制输入 u（18维关节力矩指令）的映射：
            G[0:18,  :] = 0          位置通道不受力矩直接控制
            G[18:36, :] = M^{-1}(s)  力矩→加速度（经惯性矩阵，当前用 I 占位）
            G[36:54, :] = I_18        力矩指令直接写入 tau 通道
            G[54:57, :] = 0           欧拉角不受关节力矩直接控制
            G[57:58, :] = 0           高度同上
        """
        num_envs = s.shape[0]
        G = torch.zeros((num_envs, S, U), device=s.device)

        eye_U = torch.eye(U, device=s.device).unsqueeze(0).expand(num_envs, -1, -1)

        # 力矩→加速度（待替换为真实 M^{-1}）
        # M_inv = compute_inertia_matrix_inv(s, self.robot_params)
        # G[:, 18:36, :] = M_inv
        G[:, 18:36, :] = eye_U

        # 力矩指令→tau 通道
        G[:, 36:54, :] = eye_U

        return G  # (num_envs, 58, 18)

    # -----------------------------------------------------------------------
    # 第四部分：松弛变量动力学 A(μ)
    # -----------------------------------------------------------------------

    def compute_slack_dynamics_A(self, mu: torch.Tensor) -> torch.Tensor:
        """
        松弛变量动力学对角矩阵 A(μ)，(num_envs, K, K)。

        alpha_i(mu_i) = exp(beta * mu_i) - 1
        - mu 已在 forward 中双向 clamp，保证 exp 不溢出
        - alpha 额外 clamp 到 [1e-3, 1e6]，保证 J_u 非退化且有界
        """
        alpha = torch.exp(self.beta * mu) - 1.0          # (num_envs, K)
        alpha = torch.clamp(alpha, min=1e-3, max=1e6)    # 防退化 & 防溢出双重保险
        return torch.diag_embed(alpha)                    # (num_envs, K, K)

    # -----------------------------------------------------------------------
    # 第五部分：平滑切空间基 B_u（含 Procrustes 对齐）
    # -----------------------------------------------------------------------

    def compute_smooth_tangent_basis(
        self, J_u: torch.Tensor, T_ref: torch.Tensor
    ) -> torch.Tensor:
        """
        计算平滑切空间基 B_u，满足 J_u @ B_u = 0。
        使用 Rheinboldt 移动框架算法（SVD → Procrustes 对齐 → QR 正交化）。

        Args:
            J_u  : (num_envs, K, U_ext)，U_ext = U + K = 95
            T_ref: (U_ext, U)，参考坐标系
        Returns:
            B_u  : (num_envs, U_ext, U)
        """
        num_envs = J_u.shape[0]
        B_list = []

        for i in range(num_envs):
            try:
                _, _, Vh = torch.linalg.svd(J_u[i], full_matrices=True)
            except RuntimeError:
                print(f"[ATACOM] env {i}: SVD 退化，回退到参考坐标系")
                B_list.append(T_ref.clone())
                continue

            V     = Vh.T          # (U_ext, U_ext)
            B_raw = V[:, -U:]     # 零空间基，(U_ext, U)

            # Procrustes 对齐
            M_proc           = B_raw.T @ T_ref   # (U, U)
            A_svd, _, Bh_svd = torch.linalg.svd(M_proc)
            Q_star           = A_svd @ Bh_svd    # (U, U)
            B_smooth         = B_raw @ Q_star    # (U_ext, U)

            # QR 正交化
            B_smooth, _ = torch.linalg.qr(B_smooth)
            B_list.append(B_smooth)

        return torch.stack(B_list, dim=0)  # (num_envs, U_ext, U)

    # -----------------------------------------------------------------------
    # 内部工具：鲁棒伪逆
    # -----------------------------------------------------------------------

    @staticmethod
    def _robust_pinv(J: torch.Tensor, rcond: float = 1e-4) -> torch.Tensor:
        """
        批量鲁棒伪逆。
        先尝试带 rcond 截断的批量 pinv，失败时逐环境回退到
        Tikhonov 正则化右逆：J^+ = J^T (J J^T + eps I)^{-1}

        Args:
            J     : (num_envs, K, U_ext)
        Returns:
            J_pinv: (num_envs, U_ext, K)
        """
        try:
            return torch.linalg.pinv(J, rcond=rcond)
        except Exception as e:
            print(f"[ATACOM] 批量 pinv 失败（{e}），切换逐环境正则化模式")

        num_envs, K_dim, U_ext = J.shape
        device = J.device
        dtype  = J.dtype
        J_pinv_list = []

        for i in range(num_envs):
            Ji  = J[i]          # (K, U_ext)
            JJT = Ji @ Ji.T     # (K, K)
            eps = 1e-4 * max(1.0, torch.norm(JJT).item())
            reg = eps * torch.eye(K_dim, device=device, dtype=dtype)
            try:
                # 右逆：J^+ = J^T (J J^T + eps I)^{-1}，结果 (U_ext, K)
                Ji_pinv = Ji.T @ torch.linalg.inv(JJT + reg)
            except Exception:
                # 最终兜底：lstsq 求解 Ji @ X = I，solution 形状 (U_ext, K)
                Id      = torch.eye(K_dim, device=device, dtype=dtype)
                Ji_pinv = torch.linalg.lstsq(Ji, Id).solution
            J_pinv_list.append(Ji_pinv)

        return torch.stack(J_pinv_list, dim=0)  # (num_envs, U_ext, K)

    # -----------------------------------------------------------------------
    # 第六部分：ATACOM 前向计算
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
            s            : 当前状态 (num_envs, S=58)。
                           s[:, 54:57] 为 ZYX 欧拉角 [roll, pitch, yaw]（rad）。
            u_nominal    : 名义动作，RL 策略原始输出 (num_envs, U=18)。
            ang_vel_body : 机体系角速度 [ω_x, ω_y, ω_z]（rad/s），(num_envs, 3)。
                           用于计算欧拉角漂移项 f[54:57] = T(φ) @ ω。
                           为 None 时该漂移项取 0（简化模式，保守但不精确）。

        Returns:
            u_safe : 安全动作 (num_envs, U=18)。
                     若数值崩溃（含 nan/inf）则回退到零动作。
            u_mu   : 松弛变量更新量 (num_envs, K=77)。
                     u_mu_i 表示第 i 个约束的松弛变量本步更新速率；
                     绝对值越大说明该约束被主动干预越强。
                     若含 nan/inf 则回退到零。
            info   : 约束监控信息字典，包含各类约束违反量、u_mu 分类范数等。
        """
        device   = s.device
        num_envs = s.shape[0]

        # ── 输入检查 ──────────────────────────────────────────────────────────
        _check("s [输入状态]",         s,         "Step0")
        _check("u_nominal [名义动作]", u_nominal, "Step0")
        # 修正5：补充 ang_vel_body 的数值检查，防止上游 nan 悄悄污染漂移项
        if ang_vel_body is not None:
            _check("ang_vel_body [机体角速度]", ang_vel_body, "Step0")

        # Step 1: 约束值 k(s)，(num_envs, K=77)
        k = self.compute_constraints(s)
        _check("k [约束值]", k, "Step1")
        print(f"[ATACOM Step1] k range: min={k.min().item():.3e}  max={k.max().item():.3e}")

        # 打印当前违反的约束项及其程度（k_i > 0 表示违反）
        violated = k > 0  # (num_envs, 77)
        if violated.any():
            # 约束名称表，与 compute_constraints 中 k_list 的拼装顺序严格对应
            constraint_names = []
            rp = self.robot_params
            q_max  = rp['q_max']
            q_min  = rp['q_min']
            dq_max = rp['dq_max']
            tau_max = rp['tau_max']
            for j in range(18):
                constraint_names.append(f"q[{j}]_upper  (limit={q_max[j].item():.3f})")
                constraint_names.append(f"q[{j}]_lower  (limit={q_min[j].item():.3f})")
            for j in range(18):
                constraint_names.append(f"dq[{j}]_abs   (limit=±{dq_max[j].item():.3f})")
            for j in range(18):
                constraint_names.append(f"tau[{j}]_abs  (limit=±{tau_max[j].item():.3f})")
            constraint_names.append(f"z_upper       (limit={rp['z_max']:.3f})")
            constraint_names.append(f"z_lower       (limit={rp['z_min']:.3f})")
            phi_max = rp['phi_max']
            for i, axis in enumerate(['roll', 'pitch', 'yaw']):
                constraint_names.append(f"phi_{axis}_abs (limit=±{phi_max[i].item():.3f})")

            # 取所有环境中每个约束的最大违反值，只打印确实违反的项
            k_max_per_constraint = k.max(dim=0).values  # (77,)
            print("[ATACOM Step1] 违反约束详情（显示最大违反程度）：")
            for idx, (name, val) in enumerate(zip(constraint_names, k_max_per_constraint)):
                if val.item() > 0:
                    # 找出违反该约束的环境数量
                    n_violated = violated[:, idx].sum().item()
                    print(f"  [{idx:02d}] {name:<35s}  违反量={val.item():.4e}  "
                          f"违反环境数={int(n_violated)}/{num_envs}")

        # Step 2: 松弛变量 μ
        #   下界 1e-2：避免约束违反时 mu 过小导致 A(μ) 退化
        #   上界 mu_max：保证 beta * mu < 88，防止 exp 溢出成 inf
        mu = torch.clamp(-k, min=1e-6, max=self.mu_max)   # (num_envs, K)
        _check("mu [松弛变量]", mu, "Step2")

        # Step 3: 等式约束残差 c = k + μ
        c = k + mu
        _check("c [约束残差]", c, "Step3")

        # Step 4: 约束雅可比 J_k，(num_envs, K, S)
        J_k = self.compute_constraint_jacobian(s)
        _check("J_k [约束雅可比]", J_k, "Step4")

        # Step 5: 漂移项与输入增益
        f = self.compute_drift_f(s, ang_vel_body=ang_vel_body)
        G = self.compute_input_gain_G(s)
        _check("f [漂移项]", f, "Step5")
        _check("G [输入增益]", G, "Step5")

        # Step 6: 约束漂移 ψ = J_k @ f，Drift Clipping
        psi = torch.clamp(
            torch.bmm(J_k, f.unsqueeze(-1)).squeeze(-1),
            min=0.0,
        )  # (num_envs, K)
        _check("psi [约束漂移]", psi, "Step6")

        # Step 7: 松弛变量动力学矩阵 A(μ)，(num_envs, K, K)
        A = self.compute_slack_dynamics_A(mu)
        _check("A [松弛动力学]", A, "Step7")

        # Step 8: 输入雅可比 J_u = [J_k @ G, A]，(num_envs, K, 95)
        J_G = torch.bmm(J_k, G)
        J_u = torch.cat([J_G, A], dim=2)
        _check("J_G",              J_G, "Step8")
        _check("J_u [输入雅可比]", J_u, "Step8")

        # Step 9: 平滑切空间基 B_u，(num_envs, 95, U)
        T_ref = self.T_ref.to(device)
        B_u   = self.compute_smooth_tangent_basis(J_u, T_ref)
        _check("B_u [切空间基]", B_u, "Step9")

        # Step 10: 增广名义动作 [u_nominal; 0_K]，(num_envs, 95)
        u_nom_ext = torch.cat(
            [u_nominal, torch.zeros((num_envs, K), device=device)], dim=1
        )

        # Step 11: 伪逆，(num_envs, 95, 77)
        J_u_pinv = self._robust_pinv(J_u)
        _check("J_u_pinv [伪逆]", J_u_pinv, "Step11")

        if torch.isnan(J_u_pinv).any():
            cond = torch.linalg.cond(J_u)
            print(
                f"[ATACOM Step11] J_u 条件数: "
                f"min={cond.min().item():.2e}  "
                f"max={cond.max().item():.2e}  "
                f"mean={cond.mean().item():.2e}"
            )
            J_u_pinv = torch.nan_to_num(J_u_pinv, nan=0.0, posinf=0.0, neginf=0.0)

        # 安全控制律（公式 2.39）
        term_drift       = -torch.bmm(J_u_pinv, psi.unsqueeze(-1)).squeeze(-1)
        term_contraction = -self.lam * torch.bmm(J_u_pinv, c.unsqueeze(-1)).squeeze(-1)
        term_tangential  = torch.bmm(
            B_u, torch.bmm(B_u.transpose(1, 2), u_nom_ext.unsqueeze(-1))
        ).squeeze(-1)

        _check("term_drift",       term_drift,       "Step11")
        _check("term_contraction", term_contraction, "Step11")
        _check("term_tangential",  term_tangential,  "Step11")

        u_ext = term_drift + term_contraction + term_tangential  # (num_envs, 95)
        _check("u_ext [增广安全动作]", u_ext, "Step11")

        # Step 12: 从增广安全动作中分离关节动作与松弛变量更新量
        #
        #   u_ext = [ u_safe ]  前 U=18 维：关节安全力矩指令
        #           [ u_mu   ]  后 K=77 维：松弛变量更新量 dμ/dt
        #
        # u_mu 的物理意义：
        #   ATACOM 将松弛变量 μ 视为辅助"虚拟控制输入"，与关节力矩 u 拼成增广向量。
        #   u_mu_i 表示第 i 个约束的松弛变量在本步的更新速率。
        #   其绝对值越大，说明该约束正在被主动推离边界（或被快速拉回）；
        #   正值表示 μ 在增大（约束裕量增加），负值表示 μ 在减小（约束趋紧）。
        u_safe = u_ext[:, :U]   # (num_envs, 18)
        u_mu   = u_ext[:, U:]   # (num_envs, 77)

        # ── 输出安全检查：nan/inf 回退到零动作 ────────────────────────────────
        if _check("u_safe [最终输出]", u_safe, "Step12"):
            print("[ATACOM Step12] u_safe 含 nan/inf，已回退到零动作")
            u_safe = torch.zeros_like(u_safe)

        if _check("u_mu [松弛变量更新量]", u_mu, "Step12"):
            print("[ATACOM Step12] u_mu 含 nan/inf，已回退到零")
            u_mu = torch.zeros_like(u_mu)

        # 监控信息（按约束类别分解）
        info = {
            # ── 约束状态 ────────────────────────────────────────────────────
            'constraint_value'     : k,
            'constraint_violation' : torch.clamp(k, min=0).sum(dim=1).mean().item(),
            'violation_pos'        : torch.clamp(k[:, 0:36],  min=0).sum(dim=1).mean().item(),
            'violation_vel'        : torch.clamp(k[:, 36:54], min=0).sum(dim=1).mean().item(),
            'violation_tau'        : torch.clamp(k[:, 54:72], min=0).sum(dim=1).mean().item(),
            'violation_height'     : torch.clamp(k[:, 72:74], min=0).sum(dim=1).mean().item(),
            'violation_tilt'       : torch.clamp(k[:, 74:77], min=0).sum(dim=1).mean().item(),
            # ── 松弛变量 ────────────────────────────────────────────────────
            'mu_norm'              : mu.norm(dim=1).mean().item(),
            'u_mu'                 : u_mu,                              # (num_envs, 77) 完整张量
            'u_mu_norm'            : u_mu.norm(dim=1).mean().item(),    # 标量，便于 tensorboard
            'u_mu_pos'             : u_mu[:, 0:36].norm(dim=1).mean().item(),    # 位置约束分量
            'u_mu_vel'             : u_mu[:, 36:54].norm(dim=1).mean().item(),   # 速度约束分量
            'u_mu_tau'             : u_mu[:, 54:72].norm(dim=1).mean().item(),   # 力矩约束分量
            'u_mu_height'          : u_mu[:, 72:74].norm(dim=1).mean().item(),   # 高度约束分量
            'u_mu_tilt'            : u_mu[:, 74:77].norm(dim=1).mean().item(),   # 倾角约束分量
            # ── 综合指标 ────────────────────────────────────────────────────
            'safe_ratio'           : (k <= 0).all(dim=1).float().mean().item(),
            'psi_norm'             : psi.norm(dim=1).mean().item(),
        }

        return u_safe, u_mu, info