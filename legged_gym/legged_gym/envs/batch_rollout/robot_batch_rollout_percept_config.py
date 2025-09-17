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

import os
import inspect

from legged_gym.envs.batch_rollout.robot_batch_rollout_config import RobotBatchRolloutCfg, RobotBatchRolloutCfgPPO


class RobotBatchRolloutPerceptCfg(RobotBatchRolloutCfg):
    class env(RobotBatchRolloutCfg.env):
        # Same as parent class
        pass

    # Ray caster configuration
    class raycaster:
        enable_raycast = False  # Set to True to enable ray casting
        ray_pattern = "cone"    # Options: single, grid, cone, spherical, spherical2
        num_rays = 10           # Number of rays for cone pattern
        ray_angle = 30.0        # Cone angle in degrees
        terrain_file = ""       # Path to terrain mesh file
        max_distance = 10.0     # Maximum ray cast distance
        attach_yaw_only = True  # If True, only yaw rotation is applied to rays
        offset_pos = [0.3, 0.0, 0.5]  # Offset from robot base
        # For spherical pattern
        spherical_num_azimuth = 8
        spherical_num_elevation = 4
        # For spherical2 pattern (uniform)
        spherical2_num_points = 32     # Number of points for uniform spherical distribution
        spherical2_polar_axis = [0.0, 0.0, 1.0]  # Direction of polar axis

    # SDF configuration
    class sdf:
        enable_sdf = False      # Set to True to enable SDF calculations
        mesh_paths = []         # Paths to mesh files for SDF calculation
        max_distance = 10.0     # Maximum SDF distance to compute
        enable_caching = True   # Enable SDF query caching for performance
        update_freq = 5         # Update SDF values every N steps

        # Robot body parts to compute SDF for
        # Example: ["trunk", "FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        query_bodies = []

        # Collision sphere configuration for each query body
        # collision_sphere_radius: radius of collision sphere for each query body
        # collision_sphere_pos: position of collision sphere center relative to body origin [x, y, z]
        collision_sphere_radius = []  # List of radii, one per query body
        collision_sphere_pos = []     # List of [x, y, z] offsets, one per query body

        # Whether to compute SDF gradients
        compute_gradients = True

        # Whether to compute nearest points on mesh
        compute_nearest_points = True

        # Whether to include SDF values in observations
        include_in_obs = True

    class terrain(RobotBatchRolloutCfg.terrain):
        # Same as parent class with additional field for mesh file
        # that can be used for both raycast and SDF
        mesh_file = ""  # Mesh file path for terrain

    class commands(RobotBatchRolloutCfg.commands):
        # Same as parent class
        pass

    class init_state(RobotBatchRolloutCfg.init_state):
        # Same as parent class
        pass

    class control(RobotBatchRolloutCfg.control):
        # Same as parent class
        pass

    class asset(RobotBatchRolloutCfg.asset):
        # Same as parent class
        pass

    class rewards(RobotBatchRolloutCfg.rewards):
        # Same as parent class
        pass

    class domain_rand(RobotBatchRolloutCfg.domain_rand):
        # Same as parent class
        pass

    class viewer(RobotBatchRolloutCfg.viewer):
        # Same as parent class
        pass


class RobotBatchRolloutPerceptCfgPPO(RobotBatchRolloutCfgPPO):
    # Same as parent class
    pass
