# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from isaacgym.torch_utils import quat_apply, normalize, quat_mul
import torch
from torch import Tensor
import numpy as np
from typing import Tuple

# @ torch.jit.script


def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script

def quat_apply_yaw_inverse(quat, vec):
    quat_yaw_inverse = quat.clone().view(-1, 4)
    quat_yaw_inverse[:, :2] = 0.
    quat_yaw_inverse[:, 2] = -1 * quat_yaw_inverse[:, 2]
    quat_yaw_inverse = normalize(quat_yaw_inverse)
    return quat_apply(quat_yaw_inverse, vec)

def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script


def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0., -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.) / 2.
    return (upper - lower) * r + lower


# def ypr_to_quat(yaw, pitch, roll):
#     # type: (Tensor, Tensor, Tensor) -> Tensor
#     # FIXME: Check if this is correct
#     cy = torch.cos(yaw * 0.5)
#     sy = torch.sin(yaw * 0.5)
#     cp = torch.cos(pitch * 0.5)
#     sp = torch.sin(pitch * 0.5)
#     cr = torch.cos(roll * 0.5)
#     sr = torch.sin(roll * 0.5)
#     qw = cy * cp * cr + sy * sp * sr
#     qx = cy * cp * sr - sy * sp * cr
#     qy = sy * cp * sr + cy * sp * cr
#     qz = sy * cp * cr - cy * sp * sr
#     return torch.stack([qw, qx, qy, qz], dim=-1)

def ypr_to_quat(
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    roll: torch.Tensor
) -> torch.Tensor:
    """Convert Yaw (Z), Pitch (Y), Roll (X) angles to a quaternion [x, y, z, w]."""
    # Ensure tensors are Float (to match common ML frameworks)
    yaw = yaw.float()
    pitch = pitch.float()
    roll = roll.float()

    half_angle = 0.5
    yaw_half = yaw * half_angle
    pitch_half = pitch * half_angle
    roll_half = roll * half_angle

    # Compute trigonometric components
    cy, sy = torch.cos(yaw_half), torch.sin(yaw_half)
    cp, sp = torch.cos(pitch_half), torch.sin(pitch_half)
    cr, sr = torch.cos(roll_half), torch.sin(roll_half)

    # Create individual quaternions (roll=X, pitch=Y, yaw=Z)
    q_roll = torch.stack([sr,  torch.zeros_like(sr), torch.zeros_like(sr), cr], dim=-1)   # [x, y, z, w]
    q_pitch = torch.stack([torch.zeros_like(sp), sp,  torch.zeros_like(sp), cp], dim=-1)  # [x, y, z, w]
    q_yaw = torch.stack([torch.zeros_like(sy), torch.zeros_like(sy), sy, cy], dim=-1)    # [x, y, z, w]

    # Combine: q = q_roll * q_pitch * q_yaw (applied in reverse order)
    q = quat_mul(q_roll, q_pitch)
    q = quat_mul(q, q_yaw)

    return q


################################################
# Curves
################################################


# Parameter Mat
LINEAR_MAT = torch.tensor([[1.0, 0.0], [-1.0, 1.0]], dtype=torch.float32)
BEZIER_MAT = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [-3.0, 3.0, 0.0, 0.0],
    [3.0, -6.0, 3.0, 0.0],
    [-1.0, 3.0, -3.0, 1.0]
], dtype=torch.float32)
HERMITE_MAT = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [-3.0, -2.0, 3.0, -1.0],
    [2.0, 1.0, -2.0, 1.0]
], dtype=torch.float32)
UNI_B_MAT = torch.tensor([
    [1.0, 4.0, 1.0, 0.0],
    [-3.0, 0.0, 3.0, 0.0],
    [3.0, -6.0, 3.0, 0.0],
    [-1.0, 3.0, -3.0, 1.0]
], dtype=torch.float32) / 6.0

# @torch.jit.script


def linear_evaluate(knots: torch.Tensor, t):
    """
    Evaluate linear curve at parameter t
    :param knots: control points of linear curve with shape (2, x)
    :param t: parameter tensor (scalar or 1D tensor)
    """
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=knots.dtype, device=knots.device)
    t = t.to(device=knots.device, dtype=knots.dtype).reshape(-1)
    ones = torch.ones_like(t)
    t_vec = torch.stack([ones, t], dim=1)
    linear_mat = LINEAR_MAT.to(device=knots.device, dtype=knots.dtype)
    result = t_vec @ linear_mat @ knots
    return result.flatten()

# @torch.jit.script


def cubic_evaluate(knots: torch.Tensor, t, para_mat: torch.Tensor, eval_mode="pos"):
    """
    Evaluate cubic curve at parameter t
    :param knots: control points of cubic curve with shape (4, x)
    :param t: parameter tensor (scalar or 1D tensor)
    :param para_mat: polynomial parameter matrix (4x4 torch tensor)
    :param eval_mode: "pos" or "vel"
    """
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=knots.dtype, device=knots.device)
    t = t.to(device=knots.device, dtype=knots.dtype).reshape(-1)

    if eval_mode == "pos":
        ones = torch.ones_like(t)
        t_vec = torch.stack([ones, t, t.pow(2), t.pow(3)], dim=1)
    elif eval_mode == "vel":
        zeros = torch.zeros_like(t)
        ones = torch.ones_like(t)
        t_vec = torch.stack([zeros, ones, 2*t, 3*t.pow(2)], dim=1)
    else:
        raise ValueError("Invalid eval_mode")

    para_mat = para_mat.to(device=knots.device, dtype=knots.dtype)
    point = t_vec @ para_mat @ knots
    return point.flatten() if point.size(0) == 1 else point


def cubic_bezier_evaluate(knots: torch.Tensor, t):
    """
    Evaluate cubic bezier curve at parameter t
    :param knots: control points, shape `torch.Tensor([p1, p2, p3, p4])`
    :param t: parameter tensor (scalar or 1D tensor) range from 0 to 1
    :return: point on bezier curve
    """
    return cubic_evaluate(knots, t, BEZIER_MAT)


def cubic_hermite_evaluate(knots: torch.Tensor, t):
    """
    Evaluate cubic hermite curve at parameter t
    :param knots: control points, shape `torch.Tensor([p0, v0, p1, v1])`
    :param t: parameter tensor (scalar or 1D tensor) range from 0 to 1
    :return: point on hermite curve
    """
    return cubic_evaluate(knots, t, HERMITE_MAT)

################################################
# Random Walker
################################################


class RandomWalker:
    def __init__(self,
                 bounds: torch.Tensor,
                 num_envs: int,
                 target_update_interval: float = 1.0,
                 target_track_kp: float = 1.0,
                 max_track_vel: float = 0.5,
                 distribution_type: str = 'uniform',
                 dtype=torch.float32):
        """
        Config:
        :param bounds: Tensor of shape [2, dim] - for uniform: [min, max], for normal: [mu, sigma]
        :param num_envs: Number of parallel environments
        :param target_update_interval: Time interval for target updates (seconds)
        :param max_track_vel: Maximum step velocity towards target
        :param distribution_type: 'uniform' for bounded uniform, 'normal' for normal distribution
        """
        self.dtype = dtype
        self.dim = bounds.shape[1]
        self.num_envs = num_envs
        self.bounds = bounds.to(dtype=self.dtype)
        self.target_interval = target_update_interval
        self.max_track_vel = max_track_vel
        self.distribution_type = distribution_type.lower()

        assert self.distribution_type in ['uniform', 'normal'], \
            "Invalid distribution type, choose 'uniform' or 'normal'"

        # Initialize positions and targets
        self.current_pos = self._random_positions()
        self.target_pos = self._random_positions()
        self.timers = torch.full((num_envs,), target_update_interval,
                                 dtype=self.dtype, device=bounds.device)

    def _random_positions(self):
        """Generate random positions according to configured distribution"""
        if self.distribution_type == 'uniform':
            # Original uniform distribution logic
            return torch.rand((self.num_envs, self.dim),
                              device=self.bounds.device, dtype=self.dtype) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        else:
            # Normal distribution with per-dimension parameters
            means = self.bounds[0].unsqueeze(0).expand(self.num_envs, -1)
            stds = self.bounds[1].unsqueeze(0).expand(self.num_envs, -1)
            return torch.normal(means, stds).to(dtype=self.dtype)

    def _reset_timers(self, mask):
        """Reset timers for specified environments"""
        self.timers[mask] = self.target_interval

    def step(self, dt: float):
        """
        Update walker positions
        :param dt: Time step (seconds)
        :return: Current positions [num_envs, dim]
        """
        # Update timers
        self.timers -= dt

        # Find environments needing new targets
        need_new_target = self.timers <= 0
        if torch.any(need_new_target):
            # Generate new targets for expired environments
            self.target_pos[need_new_target] = self._random_positions()[need_new_target]
            self._reset_timers(need_new_target)

        # Calculate movement towards targets
        direction = self.target_pos - self.current_pos
        step_speed = torch.clamp(direction.norm(dim=-1), max=self.max_track_vel).unsqueeze(-1)
        step_vel = direction * (step_speed / (direction.norm(dim=-1, keepdim=True) + 1e-6))

        # Update positions
        self.current_pos += step_vel * dt

        # Ensure positions stay within bounds
        if self.distribution_type == 'uniform':
            self.current_pos = torch.clamp(self.current_pos,
                                           self.bounds[0], self.bounds[1])
        return self.current_pos.clone()

    @property
    def positions(self):
        """Get current positions"""
        return self.current_pos.clone()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # # Test Curve
    # knots = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=self.dtype)
    # t = torch.linspace(0, 1, 100)
    # # curve = cubic_evaluate(knots, t, HERMITE_MAT, "pos")
    # curve = cubic_hermite_evaluate(knots, t)
    # plt.plot(curve[:, 0], curve[:, 1])
    # plt.show()

    # Test Random Walker
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    distribution_type = 'normal'  # 'uniform' or 'normal'
    if distribution_type == 'uniform':
        bounds = torch.tensor([[-5.0, -5.0], [5.0, 5.0]], device=device)  # 2D bounds
    else:
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device)
    num_envs = 64

    # Create walker
    walker = RandomWalker(
        bounds=bounds,
        num_envs=num_envs,
        target_update_interval=1.0,
        max_track_vel=1.0,
        distribution_type=distribution_type
    )

    # Simulation loop
    dt = 0.05
    plt.show(block=False)
    for _ in range(100):
        positions = walker.step(dt=dt)
        # Visualize or use positions...
        plt.clf()
        plt.scatter(positions[:, 0].cpu(), positions[:, 1].cpu())
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.pause(dt)
