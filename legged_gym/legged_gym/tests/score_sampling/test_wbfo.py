import numpy as _np
import re
import numpy as np
import matplotlib.pyplot as plt
import isaacgym
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.wbfo.wbfo import WeightedBasisFunctionOptimizer
import torch
from warp import ones

# Double integrator system dynamics (for illustration)


def double_integrator_rollout(u, x0=None):
    """
    Simulate a double integrator system given control sequence u.
    Args:
        u: [T, 1] control sequence (acceleration)
        x0: initial state [2] (position, velocity)
    Returns:
        xs: [T+1, 2] state trajectory
    """
    T = u.shape[0]
    xs = torch.zeros((T+1, 2), dtype=u.dtype)
    if x0 is not None:
        xs[0] = x0
    for t in range(T):
        xs[t+1, 1] = xs[t, 1] + u[t, 0]  # velocity update
        xs[t+1, 0] = xs[t, 0] + xs[t+1, 1]  # position update
    return xs


def single_integrator_rollout(u, x0=None):
    """
    Simulate a single integrator system given control sequence u.
    Args:
        u: [T, 1] control sequence (velocity)
        x0: initial state [2] (position, velocity)
    Returns:
        xs: [T+1, 2] state trajectory
    """
    T = u.shape[0]
    xs = torch.zeros((T+1, 2), dtype=u.dtype)
    if x0 is not None:
        xs[0] = x0
    for t in range(T):
        xs[t+1, 0] = xs[t, 0] + u[t, 0]  # position update
    return xs

# Parameters
horizon_nodes = 8
horizon_samples = 32
action_dim = 1
num_samples = 256

target_pos = 100.0  # Target position at final time
x0 = torch.tensor([0.0, 0.0])  # Initial state

# Create WBFO optimizer
wbfo = WeightedBasisFunctionOptimizer(
    horizon_nodes=horizon_nodes+1,
    horizon_samples=horizon_samples+1,
    action_dim=action_dim,
    temp_tau=0.7,
    device=torch.device('cpu')
)

# Generate initial mean trajectory (all zeros)
mean_traj = torch.zeros((horizon_nodes+1, action_dim))

# Sample trajectories around mean
noise_scale = 5.0
sampled_trajs = mean_traj + noise_scale * torch.randn((num_samples, horizon_nodes+1, action_dim))

# Convert nodes to dense controls
us_batch = wbfo.node2dense(sampled_trajs)  # [num_samples, horizon_samples+1, action_dim]

# Roll out each trajectory
xs_batch = []
for i in range(num_samples):
    xs = double_integrator_rollout(us_batch[i], x0)
    xs_batch.append(xs)
xs_batch = torch.stack(xs_batch)  # [num_samples, horizon_samples+2, 2]

# Compute rewards: negative squared distance to target at final time
final_pos = xs_batch[:, -1, 0]
rewards = -((final_pos - target_pos) ** 2).unsqueeze(1).repeat(1, horizon_samples+1)

# WBFO optimization with noise scheduling
num_diffuse_steps = 20
initial_noise_scale = noise_scale
noise_decay = 0.9
mean_history = [mean_traj.clone()]
for i in range(num_diffuse_steps):
    curr_noise = initial_noise_scale * (noise_decay ** i)
    # Resample trajectories with scheduled noise
    sampled_trajs = mean_traj + curr_noise * torch.randn_like(sampled_trajs)
    us_batch = wbfo.node2dense(sampled_trajs)
    # Roll out and compute rewards
    xs_batch = torch.stack([double_integrator_rollout(u, x0) for u in us_batch])
    final_pos = xs_batch[:, -1, 0]
    # rewards = -((final_pos - target_pos) ** 2).unsqueeze(1).repeat(1, horizon_samples+1)
    rewards = - (xs_batch[:, 1:, 0] - torch.ones_like(xs_batch[:, 1:, 0]) * target_pos) ** 2
    print(rewards)
    # Optimize mean trajectory
    mean_traj = wbfo.optimize(mean_traj, sampled_trajs, rewards)
    mean_history.append(mean_traj.clone())

updated_traj = mean_traj
updated_us = wbfo.node2dense(updated_traj)

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot all sampled trajectories (position)
for i in range(num_samples):
    xs = xs_batch[i, :, 0].numpy()
    ax1.plot(xs, color='gray', alpha=0.3)

# Plot mean trajectories over optimization steps
colors = plt.cm.viridis(_np.linspace(0, 1, len(mean_history)))
for idx, m in enumerate(mean_history):
    m_us = wbfo.node2dense(m)
    m_xs = double_integrator_rollout(m_us, x0)[:, 0].numpy()
    style = '-' if idx > 0 else '--'
    ax1.plot(m_xs, color=colors[idx], linestyle=style, alpha=0.8, 
            label=f'Mean step {idx}')
    # Also plot the control trajectory
    ax2.plot(m_us.numpy()[:, 0], color=colors[idx], linestyle=style, alpha=0.8,
            label=f'Control step {idx}')

# Plot final optimized trajectory
updated_us = wbfo.node2dense(updated_traj)
updated_xs = double_integrator_rollout(updated_us, x0)[:, 0].numpy()
ax1.plot(updated_xs, 'r-', linewidth=2, label='Final optimized')
# Plot target position
ax1.axhline(target_pos, color='g', linestyle=':', label='Target')
ax1.set_xlabel('Time step')
ax1.set_ylabel('Position')
ax1.set_title('Position Trajectories')
ax1.legend()

# Plot final optimized control
ax2.plot(updated_us.numpy()[:, 0], 'r-', linewidth=2, label='Final control')
ax2.set_xlabel('Time step')
ax2.set_ylabel('Control (acceleration)')
ax2.set_title('Control Trajectories')
ax2.legend()

plt.tight_layout()
plt.savefig('wbfo_optimization.png')  # Save figure to file
plt.show()
