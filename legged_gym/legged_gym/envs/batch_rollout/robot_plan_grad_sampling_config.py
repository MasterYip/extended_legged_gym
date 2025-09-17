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

from legged_gym.envs.batch_rollout.robot_traj_grad_sampling_config import RobotTrajGradSamplingCfg, RobotTrajGradSamplingCfgPPO


class RobotPlanGradSamplingCfg(RobotTrajGradSamplingCfg):
    """Configuration for Robot Planning Gradient Sampling Environment.

    This configuration extends RobotTrajGradSamplingCfg to add planning-specific
    settings for state velocity trajectory optimization.
    """

    class planning:
        """Planning-specific configuration parameters."""

        # State velocity dimensions for elspider robot
        # 3 (base linear velocity) + 3 (base angular velocity) + 18 (joint velocities)
        state_vel_dim = 24

        # Integration method for state rollout
        integration_method = "euler"  # "euler" or "rk4"

        # Whether to use simulation step for viewer updates
        use_sim_step_for_viewer = True

        # State velocity limits for planning
        max_base_lin_vel = 3.0  # m/s
        max_base_ang_vel = 2.0  # rad/s
        max_joint_vel = 10.0   # rad/s

        # Integration stability parameters
        max_integration_step = 0.01  # Maximum integration step size

        # Whether to enforce kinematic constraints during integration
        enforce_joint_limits = False

        # Noise scale for state velocity sampling
        state_vel_noise_scale = 1.0

    class trajectory_opt(RobotTrajGradSamplingCfg.trajectory_opt):
        """Override trajectory optimization settings for planning."""
        enable_traj_opt = True
        horizon_samples = 100
        horizon_nodes = 20
        num_samples = 64
        num_diffuse_steps = 2
        num_diffuse_steps_init = 10
        temp_sample = 0.1
        horizon_diffuse_factor = 0.98
        traj_diffuse_factor = 0.95
        update_method = "mppi"  # "mppi" or "wbfo"
        interp_method = "linear"  # "linear" or "spline"
        compute_predictions = True

        # Planning-specific: optimize state velocities instead of actions
        optimize_state_velocities = True

    class rl_warmstart(RobotTrajGradSamplingCfg.rl_warmstart):
        """Override RL warmstart settings for planning."""
        enable = False  # Disable by default for planning
        use_for_append = False
        obs_type = "privileged"  # "privileged" or "observation"
        policy_path = ""

    class rewards(RobotTrajGradSamplingCfg.rewards):
        """Override reward settings for planning."""
        # Planning rewards focus on state trajectory quality
        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time = 1.0
            collision = -1.
            action_rate = -0.01
            stand_still = -0.
            # Planning-specific rewards
            state_vel_smoothness = -0.01
            trajectory_deviation = -0.1


class RobotPlanGradSamplingCfgPPO(RobotTrajGradSamplingCfgPPO):
    pass
