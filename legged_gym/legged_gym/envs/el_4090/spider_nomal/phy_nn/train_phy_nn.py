
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
import os
import random
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Parameter summary and file-level notes

K_e 和 K_t 在本实现中作为常数使用（都设为 2.1），不作为可学习参数。

可辨识物理参数（`phy_theta` 向量，顺序固定）：
 0: R_a    -- 电枢电阻 (Ohm)
 1: L_a    -- 电枢电感 (H)
 2: N      -- 减速比（齿比），电机轴角速度 = N * 负载角速度
 3: tau_c  -- 库仑摩擦 (Nm)
 4: B_v    -- 粘性摩擦系数 (Nm·s/rad)
 5: tau_s  -- Stribeck 深度 (Nm)
 6: omega_s-- Stribeck 临界速度 (rad/s)
 7: k2     -- 位置比例项（用于产生等效电流，单位 A/rad）
 8: k3     -- 速度微分项（用于产生等效电流，单位 A/(rad/s)）
 9: k_g    -- 重力项系数（在电流预测中以加法项出现，用于表征重力产生的等效电流）

残差网络参数：`ResidualNN` 的所有权重与偏置，用于拟合物理模型与观测之间的剩余误差。

数据与输入说明：
- CSV 字段 `pos_cmd`, `pos_fdb`, `dof_vel` 各为 18 维数组，对应 18 个电机通道。
- 程序先按 `new_order` 重排通道顺序（映射见代码中的 `new_order` 列表），然后对每个时间步和每个关节构造样本：
    输入: [i_curr, q_des, q_curr, q_curr_vel, joint_idx]
    目标: 下一时刻的观测值（代码中使用 `pos_fdb` 作为电流占位，若有真实电流请替换）

可选行为：命令行参数 `--joint` 用于只选择重排后索引为 k 的关节进行辨识；默认值 -1 表示使用全部 18 个关节的数据。

注意：物理模型中对驱动电压的近似为 `V_a = k_cmd * (q_des - q_curr)`，以及时间步 `dt` 默认为 0.002，可按需调整。
"""

# 物理模型部分
def tau_f_stribeck(vel, tau_c, B_v, tau_s, omega_s):
    """
    Stribeck 分段摩擦模型，支持 torch Tensor 或 numpy array（已用于 torch 运算）。
    返回与 vel 同形状的摩擦力矩（可为正负）。
    """
    # 使用 torch if possible, otherwise fall back to numpy
    import torch as _th
    is_tensor = isinstance(vel, _th.Tensor)

    if is_tensor:
        sgn = _th.sign(vel)
        absvel = _th.abs(vel)
        cond = absvel > omega_s
        term1 = tau_c * sgn + (B_v + tau_s * _th.exp(-absvel / omega_s)) * vel
        term2 = tau_c * sgn + B_v * vel + tau_s * (vel / omega_s)
        return _th.where(cond, term1, term2)
    else:
        import numpy as _np
        sgn = _np.sign(vel)
        absvel = _np.abs(vel)
        cond = absvel > omega_s
        term1 = tau_c * sgn + (B_v + tau_s * _np.exp(-absvel / omega_s)) * vel
        term2 = tau_c * sgn + B_v * vel + tau_s * (vel / omega_s)
        return _np.where(cond, term1, term2)


def phymodel(i_curr, q_des, q_curr, q_curr_vel, phy_theta, dt=0.002, k_cmd=1.0):
    """
    离散化的电气-机械耦合模型（预测下一时刻电流）：

    电气： V_a = R_a * I_a + L_a dI/dt + K_e * omega_m
    电磁力矩： tau_m = K_t * I_a
    机械（用于摩擦映射）： tau_m = N * (J_eq*ddtheta + B_eq*dtheta + tau_f)

    为保持与数据接口兼容，使用 q_des,q_curr,q_curr_vel 估算驱动电压：
    V_a ≈ k_cmd * (q_des - q_curr)

    返回预测的下一个电流值（与训练目标格式一致）。
    """
    # constants
    K_e_const = 2.1
    K_t_const = 2.1

    # phy_theta: [R_a, L_a, N, tau_c, B_v, tau_s, omega_s, k2, k3, k_g]
    R_a, L_a, N, tau_c, B_v, tau_s, omega_s, k2, k3, k_g = phy_theta

    # motor angular velocity (motor shaft) = N * load velocity
    omega_m = N * q_curr_vel

    # approximate actuator voltage from position error (simple proportional controller)
    V_a = k_cmd * (q_des - q_curr)

    # electrical dynamics: dI = (V_a - R_a * I - K_e * omega_m) / L_a
    dI = (V_a - R_a * i_curr - K_e_const * omega_m) / L_a
    i_next = i_curr + dt * dI

    # compute friction current needed to overcome load-side摩擦: I_f = (N * tau_f) / K_t
    tau_f = tau_f_stribeck(q_curr_vel, tau_c, B_v, tau_s, omega_s)
    I_f = (N * tau_f) / K_t_const

    # PD-like feed terms converted to equivalent current (k2, k3)
    I_pd = (k2 / K_t_const) * (q_des - q_curr) - (k3 / K_t_const) * q_curr_vel

    # 加上重力项对电流的等效贡献 k_g
    # k_g 假设为标量并广播到输入形状
    return i_next + I_f + I_pd + k_g

# 数据读取：使用相对于仓库根的确定路径（对从项目根执行脚本友好）
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
csv_path = os.path.join(repo_root, 'logs', 'el4090', '1', 'xy_format', 'motor_rl_xy.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")
df = pd.read_csv(csv_path).astype(str)
new_order = [15, 16, 17, 9, 10, 11, 12, 13, 14, 6, 7, 8, 0, 1, 2, 3, 4, 5]

i_curr_all = []
data_list = []
for _, row in df.iterrows():
    pos_cmd = np.array([float(val) for val in row['pos_cmd'].strip('()').split(', ')])[new_order]
    pos_fdb = np.array([float(val) for val in row['pos_fdb'].strip('()').split(', ')])[new_order]
    dof_vel = np.array([float(val) for val in row['dof_vel'].strip('()').split(', ')])[new_order]
    i_curr_all.append(pos_fdb)  # 如有真实电流数据请替换为电流
    data_list.append({
        'q_des': pos_cmd,
        'q_curr': pos_fdb,
        'q_curr_vel': dof_vel
    })

inputs = []
targets = []
num_joints = len(data_list[0]['q_curr'])
for t in range(1, len(data_list)):
    d_now = data_list[t]
    i_curr_vec = i_curr_all[t-1]
    i_next_vec = i_curr_all[t]
    for j in range(num_joints):
        inputs.append([
            i_curr_vec[j], d_now['q_des'][j], d_now['q_curr'][j], d_now['q_curr_vel'][j], j
        ])
        targets.append(i_next_vec[j])
inputs = torch.tensor(inputs, dtype=torch.float)
targets = torch.tensor(targets, dtype=torch.float)

# 数据分割：前80%用于训练和测试，后20%用于泛化测试
data_size = inputs.shape[0]
train_test_size = int(data_size * 0.8)
train_test_inputs = inputs[:train_test_size]
train_test_targets = targets[:train_test_size]
generalization_inputs = inputs[train_test_size:]
generalization_targets = targets[train_test_size:]
print(f"总样本数: {data_size}, 训练/测试集: {train_test_size}, 泛化测试集: {data_size - train_test_size}")

# 训练/测试集分割（80/20 of first 80%）
indices = list(range(train_test_inputs.shape[0]))
random.seed(42)
random.shuffle(indices)
split = int(len(indices) * 0.8)
train_idx, test_idx = indices[:split], indices[split:]
train_inputs, train_targets = train_test_inputs[train_idx], train_test_targets[train_idx]
test_inputs, test_targets = train_test_inputs[test_idx], train_test_targets[test_idx]
print(f"训练集: {len(train_idx)}, 测试集: {len(test_idx)}")

# 物理参数可学习


# 命令行参数：使用 `--test` 切换到测试模式，`--epochs` 设置训练轮数
parser = argparse.ArgumentParser(description='Train or test physical+residual NN')
parser.add_argument('--test', action='store_true', help='Run in test mode (load models and evaluate)')
parser.add_argument('--epochs', type=int, default=3000, help='Number of training epochs')
parser.add_argument('--joint', type=int, default=-1, help='If >=0, only use this (reordered) joint index 0..17 for identification')
parser.add_argument('--acc-tol', type=float, default=0.01, help='Absolute error tolerance for accuracy metric')
parser.add_argument('--plot-out', type=str, default=os.path.join(repo_root, 'results', 'pictures', 'train_phy_nn_overfit_curve.png'), help='Output path for loss/accuracy curve')
parser.add_argument('--metrics-out', type=str, default=os.path.join(repo_root, 'results', 'data', 'train_phy_nn_metrics.csv'), help='Output path for per-epoch metrics CSV')
args = parser.parse_args()
TEST = args.test
EPOCHS = args.epochs
SELECT_JOINT = args.joint if args.joint >= 0 else None
ACC_TOL = args.acc_tol

# 如果指定了单个关节索引，则过滤训练/测试集只保留该关节的数据
if SELECT_JOINT is not None:
    if SELECT_JOINT < 0 or SELECT_JOINT >= num_joints:
        raise ValueError(f"--joint must be in [0, {num_joints-1}] or -1 for all, got {SELECT_JOINT}")
    train_mask = train_inputs[:, 4] == float(SELECT_JOINT)
    test_mask = test_inputs[:, 4] == float(SELECT_JOINT)
    train_inputs = train_inputs[train_mask]
    train_targets = train_targets[train_mask]
    test_inputs = test_inputs[test_mask]
    test_targets = test_targets[test_mask]
    print(f"已选择关节 {SELECT_JOINT} 进行辨识，训练样本: {train_inputs.shape[0]}, 测试样本: {test_inputs.shape[0]}")
else:
    print('使用全部关节数据进行辨识')

# learnable physical parameters:
# [R_a, L_a, N, tau_c, B_v, tau_s, omega_s, k2, k3, k_g]
phy_theta = torch.nn.Parameter(torch.tensor([
    0.5,    # R_a (ohm)
    0.01,   # L_a (H)
    9.0,   # N (gear ratio)
    0.1,    # tau_c (Coulomb)
    0.01,   # B_v (viscous)
    0.2,    # tau_s (Stribeck depth)
    0.05,   # omega_s (Stribeck velocity)
    5.0,    # k2 (position proportional current term)
    0.1,    # k3 (velocity differential current term)
    0.2     # k_g (gravity current term, initial 0)
], dtype=torch.float), requires_grad=True)

# 神经网络残差部分
class ResidualNN(nn.Module):
    def __init__(self, input_dim=5, hidden=32, layers=2):
        super().__init__()
        mods = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(layers-1):
            mods += [nn.Linear(hidden, hidden), nn.ReLU()]
        mods += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*mods)
    def forward(self, x):
        return self.net(x).squeeze(-1)


residual_nn = ResidualNN()
optimizer = Adam([{'params': [phy_theta]}, {'params': residual_nn.parameters()}], lr=1e-2)


def evaluate_split(model_theta, model_nn, split_inputs, split_targets, tol):
    with torch.no_grad():
        phy_out = phymodel(split_inputs[:, 0], split_inputs[:, 1], split_inputs[:, 2], split_inputs[:, 3], model_theta)
        nn_out = model_nn(split_inputs)
        pred = phy_out + nn_out
        diff = pred - split_targets
        loss = ((diff)**2).mean().item()
        mae = torch.abs(diff).mean().item()
        acc = (torch.abs(diff) <= tol).float().mean().item()
    return loss, mae, acc, pred, diff


def test_model(phy_theta, residual_nn, inputs, targets):
    with torch.no_grad():
        tol = ACC_TOL if 'ACC_TOL' in globals() else 0.01
        _, _, acc, pred, diff = evaluate_split(phy_theta, residual_nn, inputs, targets, tol)
        print('--- 测试结果 ---')
        for i in range(min(10, len(targets))):
            print(f"样本{i}: 真实值={targets[i].item():.4f}, 预测值={pred[i].item():.4f}, 差值={diff[i].item():.4f}")
        print(f"均方误差: {((pred - targets) ** 2).mean().item():.6f}, 准确率(±{tol}): {acc:.4f}")


def save_overfit_diagnosis_plot(history, save_path):
    if not history:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = [item['epoch'] for item in history]
    train_loss = [item['train_loss'] for item in history]
    test_loss = [item['test_loss'] for item in history]
    gen_loss = [item['gen_loss'] for item in history]
    train_acc = [item['train_acc'] for item in history]
    test_acc = [item['test_acc'] for item in history]
    gen_acc = [item['gen_acc'] for item in history]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(epochs, train_loss, label='train_loss', linewidth=1.5)
    axes[0].plot(epochs, test_loss, label='test_loss', linewidth=1.5)
    axes[0].plot(epochs, gen_loss, label='generalization_loss', linewidth=1.5)
    axes[0].set_ylabel('MSE loss')
    axes[0].set_title('Overfitting diagnosis: loss curves')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label='train_acc', linewidth=1.5)
    axes[1].plot(epochs, test_acc, label='test_acc', linewidth=1.5)
    axes[1].plot(epochs, gen_acc, label='generalization_acc', linewidth=1.5)
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel(f'accuracy (|err| <= {ACC_TOL})')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'过拟合诊断曲线已保存到 {save_path}')


# Save models under scripts/model/phy+nn (sibling to this train folder)
# script is in scripts/train/, so go up one level to scripts/
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'phy+nn'))
os.makedirs(model_dir, exist_ok=True)
phy_theta_path = os.path.join(model_dir, 'phy_theta.pt')
nn_path = os.path.join(model_dir, 'residual_nn.pt')

if TEST:
    # 测试模式，自动加载参数（更安全，仅加载state_dict）
    if os.path.exists(phy_theta_path) and os.path.exists(nn_path):
        loaded = torch.load(phy_theta_path, map_location='cpu')
        # 兼容多种历史格式，目标为新格式长度 10:
        # [R_a, L_a, N, tau_c, B_v, tau_s, omega_s, k2, k3, k_g]
        if isinstance(loaded, torch.Tensor):
            n = loaded.numel()
            if n == 10:
                phy_theta.data = loaded.clone().detach()
            elif n == 9:
                # 旧格式包含 K_e,K_t -> remap indices to new (插入 k2,k3=0)
                old = loaded
                new = torch.tensor([old[0].item(), old[1].item(), old[4].item(), old[5].item(), old[6].item(), old[7].item(), old[8].item(), 0.0, 0.0, 0.0], dtype=torch.float)
                phy_theta.data = new
            elif n == 8:
                # 旧格式为之前的 8 元素（无 k2,k3），插入 k2,k3=0
                old = loaded
                new = torch.tensor([old[0].item(), old[1].item(), old[2].item(), old[3].item(), old[4].item(), old[5].item(), old[6].item(), 0.0, 0.0, old[7].item()], dtype=torch.float)
                phy_theta.data = new
            else:
                # 无法识别长度，尝试转为 tensor 并截断/填充
                arr = torch.tensor(loaded, dtype=torch.float)
                if arr.numel() < 10:
                    tmp = torch.zeros(10, dtype=torch.float)
                    tmp[:arr.numel()] = arr
                    phy_theta.data = tmp
                else:
                    phy_theta.data = arr[:10]
        else:
            phy_theta.data = torch.tensor(loaded, dtype=torch.float)
        residual_nn.load_state_dict(torch.load(nn_path, map_location='cpu'))
        print('已加载参数，进行测试集评估:')
        test_model(phy_theta, residual_nn, test_inputs, test_targets)
        print('泛化测试（后20%数据）：')
        test_model(phy_theta, residual_nn, generalization_inputs, generalization_targets)
    else:
        print('未找到参数文件，无法测试')
else:
    # 训练
    epochs = EPOCHS
    history = []
    best_test_loss = float('inf')
    best_test_epoch = -1
    for epoch in range(epochs):
        phy_out = phymodel(train_inputs[:, 0], train_inputs[:, 1], train_inputs[:, 2], train_inputs[:, 3], phy_theta)
        pred = phy_out + residual_nn(train_inputs)
        loss = ((pred - train_targets) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            tol = ACC_TOL if 'ACC_TOL' in globals() else 0.01
            train_loss, _, acc_tr, _, _ = evaluate_split(phy_theta, residual_nn, train_inputs, train_targets, tol)
            test_loss, _, acc_t, _, _ = evaluate_split(phy_theta, residual_nn, test_inputs, test_targets, tol)
            gen_loss, _, acc_g, _, _ = evaluate_split(phy_theta, residual_nn, generalization_inputs, generalization_targets, tol)
            gap = test_loss - train_loss
            print(
                f"epoch {epoch}, train_loss {train_loss:.6f}, test_loss {test_loss:.6f}, gen_loss {gen_loss:.6f}, "
                f"gap(test-train) {gap:.6f}, R_a={phy_theta[0].item():.4f}, L_a={phy_theta[1].item():.4f}, N={phy_theta[2].item():.4f}, k2={phy_theta[7].item():.4f}, k3={phy_theta[8].item():.4f}, k_g={phy_theta[9].item():.6f}"
            )
            print(f"准确率(阈值±{tol}): 训练={acc_tr:.4f}, 测试={acc_t:.4f}, 泛化={acc_g:.4f}")
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'gen_loss': gen_loss,
                'train_acc': acc_tr,
                'test_acc': acc_t,
                'gen_acc': acc_g,
            })
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_epoch = epoch
    print('优化完成，最终物理参数:', phy_theta.data.tolist())
    # 保存参数（更安全，仅保存state_dict）
    torch.save(phy_theta.data.clone(), phy_theta_path)
    torch.save(residual_nn.state_dict(), nn_path)
    print(f'物理参数已保存到 {phy_theta_path}')
    print(f'神经网络参数已保存到 {nn_path}')
    # 训练集评估
    print('训练集评估:')
    test_model(phy_theta, residual_nn, train_inputs, train_targets)
    # 测试集评估
    print('测试集评估:')
    test_model(phy_theta, residual_nn, test_inputs, test_targets)
    # 泛化测试（后20%数据）
    print('泛化测试（后20%数据）：')
    test_model(phy_theta, residual_nn, generalization_inputs, generalization_targets)
    # 打印最终准确率汇总
    with torch.no_grad():
        tol = ACC_TOL if 'ACC_TOL' in globals() else 0.01
        pred_tr = phymodel(train_inputs[:,0], train_inputs[:,1], train_inputs[:,2], train_inputs[:,3], phy_theta) + residual_nn(train_inputs)
        pred_te = phymodel(test_inputs[:,0], test_inputs[:,1], test_inputs[:,2], test_inputs[:,3], phy_theta) + residual_nn(test_inputs)
        pred_gen = phymodel(generalization_inputs[:,0], generalization_inputs[:,1], generalization_inputs[:,2], generalization_inputs[:,3], phy_theta) + residual_nn(generalization_inputs)
        acc_tr_final = (torch.abs(pred_tr - train_targets) <= tol).float().mean().item()
        acc_te_final = (torch.abs(pred_te - test_targets) <= tol).float().mean().item()
        acc_gen_final = (torch.abs(pred_gen - generalization_targets) <= tol).float().mean().item()
    print(f'最终准确率(阈值±{tol}): 训练={acc_tr_final:.4f}, 测试={acc_te_final:.4f}, 泛化={acc_gen_final:.4f}')
    print(f'最优测试集损失出现在 epoch {best_test_epoch}，best_test_loss={best_test_loss:.6f}')

    if history:
        metrics_df = pd.DataFrame(history)
        os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
        metrics_df.to_csv(args.metrics_out, index=False)
        print(f'每轮诊断数据已保存到 {args.metrics_out}')
        save_overfit_diagnosis_plot(history, args.plot_out)
