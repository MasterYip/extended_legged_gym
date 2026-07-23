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

import os
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
import torch
import numpy as np

from isaacgym.torch_utils import *
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.terrain_obj import TerrainObj
from legged_gym.utils.terrain_confine import TerrainConfined
from legged_gym.utils.math_utils import *
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.env_replay_mixin import RecordReplayMixin
from .robot_batch_rollout_rew_mixin import RobotBatchRolloutRewMixin


class RobotBatchRollout(BaseTask, RobotBatchRolloutRewMixin, RecordReplayMixin):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.debug_viz_origins = False  # Flag to control visualization of origins
        self.init_done = False

        # Main environments vs rollout environments
        self.num_main_envs = self.cfg.env.num_envs
        self.num_rollout_per_main = self.cfg.env.rollout_envs

        # Total number of envs = main_envs + (main_envs * rollout_envs)
        # This is the real number of envs in the simulation
        self.total_num_envs = self.num_main_envs * (1 + self.num_rollout_per_main)

        # NOTE:Temporarily modify cfg.env.num_envs to create all environments
        # Store original num_envs to restore later after setup
        self.original_num_envs = self.cfg.env.num_envs
        self.cfg.env.num_envs = self.total_num_envs

        self._parse_cfg(self.cfg)
        # Call parent class constructor with modified config
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        RobotBatchRolloutRewMixin.__init__(self)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        # Initialize buffers for main and rollout envs
        self._init_buffers()
        self._prepare_reward_function()

        # Restore original num_envs
        self.cfg.env.num_envs = self.original_num_envs
        self.num_envs = self.original_num_envs
        self.init_done = True

        # For EMA of acceleration
        self.acc_ema = 0.9
        
        # Initialize time counters for main and rollout environments
        self.t_main = 0.0
        self.t_rollout = 0.0

        self.record_replay_mixin_init()

        if not self.headless and self.cfg.viewer.ref_env is not None:
            # If a reference environment is specified, update the camera position
            ref_env_idx = self.cfg.viewer.ref_env
            pos = self.cfg.viewer.pos
            lookat = self.cfg.viewer.lookat
            if ref_env_idx is not None and ref_env_idx < self.num_main_envs:
                self.set_camera(self.env_origins[ref_env_idx] + torch.tensor([pos[0], pos[1], pos[2]], device=self.device),
                                self.env_origins[ref_env_idx] + torch.tensor([lookat[0], lookat[1], lookat[2]], device=self.device))


    # === Setups === #

    def _init_env_indices(self):
        """Initialize indices for main environments and rollout environments"""
        # Indices for main environments (the ones we actually control)
        # Every (1 + num_rollout_per_main) environment is a main environment
        self.main_env_indices = torch.arange(
            0,
            self.total_num_envs,
            1 + self.num_rollout_per_main,
            device=self.device
        )

        # Create a mapping from rollout envs to their corresponding main env
        # For rollout env i, its main env is at index main_env_indices[i // self.num_rollout_per_main]
        self.rollout_to_main_map = torch.zeros(self.total_num_envs, dtype=torch.long, device=self.device)
        self.is_main_env = torch.zeros(self.total_num_envs, dtype=torch.bool, device=self.device)
        self.is_rollout_env = torch.zeros(self.total_num_envs, dtype=torch.bool, device=self.device)

        # Mark which environments are main vs rollout
        self.is_main_env[self.main_env_indices] = True
        self.is_rollout_env = ~self.is_main_env

        # For each environment, compute its main env index
        for i in range(self.num_main_envs):
            main_idx = self.main_env_indices[i]
            start_rollout = main_idx + 1
            end_rollout = start_rollout + self.num_rollout_per_main

            # Main env maps to itself
            self.rollout_to_main_map[main_idx] = main_idx

            # Rollout envs map to their main env
            if start_rollout < self.total_num_envs:
                indices = torch.arange(start_rollout, min(end_rollout, self.total_num_envs), device=self.device)
                self.rollout_to_main_map[indices] = main_idx

        # Get rollout env indices
        self.rollout_env_indices = torch.nonzero(self.is_rollout_env).flatten()

        # For each main env, get its rollout env indices
        self.main_to_rollout_indices = []
        for i in range(self.num_main_envs):
            main_idx = self.main_env_indices[i]
            rollout_indices = torch.nonzero(self.rollout_to_main_map == main_idx).flatten()
            # Remove the main env itself from the list
            rollout_indices = rollout_indices[rollout_indices != main_idx]
            self.main_to_rollout_indices.append(rollout_indices)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0.  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities """
        # Get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Get rigid body state tensor
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Create wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.total_num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.total_num_envs, self.num_dof, 2)[..., 1]
        self.base_pos = self.root_states[:, :3]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.total_num_envs, -1, 3
        )
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)

        # Initialize data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        # This gravity is normalized (len = 1)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.total_num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.total_num_envs, 1))

        self.torques = torch.zeros(
            self.total_num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(
            self.total_num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.total_num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.commands = torch.zeros(
            self.total_num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device, requires_grad=False,
        )

        # Foot state
        self.feet_air_time = torch.zeros(
            self.total_num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False
        )
        self.feet_contact_time = torch.zeros(
            self.total_num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_contacts = torch.zeros(
            self.total_num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )

        self.foot_positions = self.rigid_body_state.view(self.total_num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.foot_velocities = self.rigid_body_state.view(self.total_num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        # Base state
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_lin_acc = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10] - self.last_root_vel[:, :3]
        ) / self.dt
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_ang_acc = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13] - self.last_root_vel[:, 3:]
        ) / self.dt
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # Joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False

            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True

            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _process_rigid_shape_props(self, props, env_id):
        """ Process rigid shape properties. Override as needed. """
        # This is the same as in LeggedRobot
        # TODO: rollout envs should keep the same friction as their main env
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # Prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.total_num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Process DOF properties. Override as needed. """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()

                # Soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        """ Process rigid body properties. Override as needed. """
        # Randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        # Convert height samples to tensor
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(
            self.terrain.tot_rows, self.terrain.tot_cols
        ).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution

        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order='C'),
            self.terrain.triangles.flatten(order='C'),
            tm_params
        )

        self.height_samples = torch.tensor(self.terrain.heightsamples).view(
            self.terrain.tot_rows, self.terrain.tot_cols
        ).to(self.device)

    def _create_envs(self):
        """ Creates environments:
            1. Loads the robot URDF/MJCF asset
            2. For each environment:
               2.1 Creates the environment
               2.2 Calls DOF and Rigid shape properties callbacks
               2.3 Creates actor with these properties and adds them to the env
            3. Stores indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # Save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)

        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)

        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])

        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        for i in range(self.total_num_envs):
            # Create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.total_num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0
            )

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # Create tensors for feet, penalized contacts, and termination contacts
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

    def create_sim(self):
        """ Creates simulation, terrain and environments """
        self._init_env_indices()

        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            if self.cfg.terrain.use_terrain_obj:
                self.terrain = TerrainObj(self.cfg.terrain)
            else:
                self.terrain = Terrain(self.cfg.terrain, self.total_num_envs)
        elif mesh_type == 'confined_trimesh':
            self.terrain = TerrainConfined(self.cfg.terrain, self.total_num_envs)

        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type in ['trimesh', 'confined_trimesh']:
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh, confined_trimesh]")

        self._create_envs()

    # === Step === #

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_main_envs, num_actions_per_env)
        """

        # Actions are provided for main environments only
        # We need to expand them for all environments
        full_actions = torch.zeros((self.total_num_envs, self.num_actions), device=self.device)

        # Copy actions for main environments
        full_actions[self.main_env_indices] = actions

        # Apply clipping to all actions
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(full_actions, -clip_actions, clip_actions).to(self.device)

        # Reset rollout environments to the state of their main environment
        self._sync_main_to_rollout()
        # Step physics and render each frame
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        self.render()
        self.record_step()

        # Return observations, privileged observations, rewards, dones and infos
        # for the main environments only
        clip_obs = self.cfg.normalization.clip_observations

        # Main environment observations
        main_obs = torch.clip(self.obs_buf[self.main_env_indices], -clip_obs, clip_obs)

        # Main environment privileged observations (if used)
        main_privileged_obs = None
        if self.privileged_obs_buf is not None:
            main_privileged_obs = torch.clip(
                self.privileged_obs_buf[self.main_env_indices], -clip_obs, clip_obs
            )

        # Main environment rewards, resets, and extras
        main_rew = self.rew_buf[self.main_env_indices]
        main_reset = self.reset_buf[self.main_env_indices]
        main_extras = {key: val[self.main_env_indices] for key, val in self.extras.items() if isinstance(val, torch.Tensor)}

        # Add non-tensor extras
        for key, val in self.extras.items():
            if not isinstance(val, torch.Tensor):
                main_extras[key] = val

        # Cache the state of main environments so they can be restored after rollouts
        self._cache_main_env_states()
        # Reset rollout environments to the state of their main environment
        self._sync_main_to_rollout()

        # Update time counters for main environments
        self.t_main += self.dt
        self.t_rollout = self.t_main

        return main_obs, main_privileged_obs, main_rew, main_reset, main_extras

    def step_rollout(self, rollout_actions, noise_scales=None):
        """ Apply actions to rollout environments with optional Gaussian noise.
            Supports two interfaces:
            1. New: Direct batch of actions for all rollout envs (num_main_envs * num_rollout_envs, num_actions_per_env)
            2. Legacy: Mean actions per main env (num_main_envs, num_actions_per_env) with noise_scales

        Args:
            rollout_actions (torch.Tensor): Either:
                - Full actions tensor of shape (num_rollout_envs, num_actions_per_env)
                - Mean action tensor of shape (num_main_envs, num_actions_per_env) [legacy]
            noise_scales (torch.Tensor, optional): Scale of noise to apply to actions, shape (num_actions_per_env).
                Only used in legacy mode.

        Returns:
            tuple: (rollout_obs, rollout_privileged_obs, rollout_rew, rollout_reset, rollout_extras)
        """
        # Determine the mode based on the input shape
        is_legacy_mode = (rollout_actions.shape[0] == self.num_main_envs)

        if is_legacy_mode:
            # Legacy mode - using mean actions per main env
            rollout_action_mean = rollout_actions

            # Create actions for rollout environments - start with the provided action means
            actions = torch.zeros((len(self.rollout_env_indices), self.num_actions), device=self.device)

            # For each rollout environment, apply the action from its main environment and add noise
            for i, rollout_idx in enumerate(self.rollout_env_indices):
                main_idx = self.rollout_to_main_map[rollout_idx]
                main_env_local_idx = torch.nonzero(self.main_env_indices == main_idx).item()

                # Get mean action from its main environment
                mean_action = rollout_action_mean[main_env_local_idx]

                # Apply Gaussian noise with specified noise_scales if provided
                if noise_scales is not None:
                    noise = torch.randn_like(mean_action, device=self.device) * noise_scales
                    actions[i] = mean_action + noise
                else:
                    actions[i] = mean_action
        else:
            # New mode - direct actions for all rollout envs
            actions = rollout_actions

            # Verify shape
            if actions.shape[0] != len(self.rollout_env_indices):
                raise ValueError(f"Expected actions shape ({len(self.rollout_env_indices)}, {self.num_actions}), "
                                 f"got {actions.shape}")

        # Apply clipping to all actions
        clip_actions = self.cfg.normalization.clip_actions
        actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # Set actions for rollout environments
        self.actions[self.rollout_env_indices] = actions

        # Step physics and render each frame for rollout environments only
        if self.cfg.viewer.render_rollouts:
            self.render()
        for _ in range(self.cfg.control.decimation):
            # Compute torques for rollout environments only
            rollout_torques = self._compute_torques(self.actions, self.rollout_env_indices)
            self.torques[self.rollout_env_indices] = rollout_torques

            # Apply torques to rollout environments
            env_ids_int32 = self.rollout_env_indices.to(dtype=torch.int32)
            self.gym.set_dof_actuation_force_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.torques),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32)
            )

            # Simulate rollout environments
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)

            # Refresh state tensors for rollout environments
            self.gym.refresh_dof_state_tensor(self.sim)

        # Process post-physics step only for rollout environments
        self.post_physics_step_rollout()

        # Restore the main environments to their cached state (freeze them)
        self._restore_main_env_states()

        # Clip observations
        clip_obs = self.cfg.normalization.clip_observations
        rollout_obs = torch.clip(self.obs_buf[self.rollout_env_indices], -clip_obs, clip_obs)

        # Get privileged observations if available
        rollout_privileged_obs = None
        if self.privileged_obs_buf is not None:
            rollout_privileged_obs = torch.clip(
                self.privileged_obs_buf[self.rollout_env_indices], -clip_obs, clip_obs
            )

        # Get rewards and reset flags
        rollout_rew = self.rew_buf[self.rollout_env_indices]
        rollout_reset = self.reset_buf[self.rollout_env_indices]

        # Prepare extras
        rollout_extras = {key: val[self.rollout_env_indices] for key, val in self.extras.items()
                          if isinstance(val, torch.Tensor)}

        # Add non-tensor extras
        for key, val in self.extras.items():
            if not isinstance(val, torch.Tensor):
                rollout_extras[key] = val

        # Update time counters for rollout environments
        self.t_rollout += self.dt

        return rollout_obs, rollout_privileged_obs, rollout_rew, rollout_reset, rollout_extras

    def post_physics_step(self):
        """ Check terminations, compute observations and rewards.
            Calls self._post_physics_step_callback() for common computations.
            Calls self._draw_debug_vis() if needed.
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Prepare quantities
        self.base_pos[:] = self.root_states[:, :3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_lin_acc[:] = self.base_lin_acc[:] * self.acc_ema + (1 - self.acc_ema) * \
            quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_ang_acc[:] = self.base_ang_acc[:] * self.acc_ema + (1 - self.acc_ema) * \
            quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13] - self.last_root_vel[:, 3:]) / self.dt
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.foot_positions = self.rigid_body_state.view(self.total_num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.foot_velocities = self.rigid_body_state.view(self.total_num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self._post_physics_step_callback()

        # Compute observations, rewards, resets
        self.check_termination()
        self.compute_reward()

        # Reset envs that need resetting
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def post_physics_step_rollout(self):
        """ Process post-physics steps for rollout environments only.
            Similar to post_physics_step but with optimizations for rollout envs.
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Update state tensors for rollout environments only
        rollout_indices = self.rollout_env_indices

        # Prepare quantities for rollout environments
        self.base_pos[rollout_indices] = self.root_states[rollout_indices, :3]
        self.base_quat[rollout_indices] = self.root_states[rollout_indices, 3:7]
        self.base_lin_vel[rollout_indices] = quat_rotate_inverse(
            self.base_quat[rollout_indices],
            self.root_states[rollout_indices, 7:10]
        )
        self.base_lin_acc[rollout_indices] = self.base_lin_acc[rollout_indices] * self.acc_ema + (1 - self.acc_ema) * \
            quat_rotate_inverse(
                self.base_quat[rollout_indices],
                self.root_states[rollout_indices, 7:10] - self.last_root_vel[rollout_indices, :3]
        ) / self.dt
        self.base_ang_vel[rollout_indices] = quat_rotate_inverse(
            self.base_quat[rollout_indices],
            self.root_states[rollout_indices, 10:13]
        )
        self.base_ang_acc[rollout_indices] = self.base_ang_acc[rollout_indices] * self.acc_ema + (1 - self.acc_ema) * \
            quat_rotate_inverse(
                self.base_quat[rollout_indices],
                self.root_states[rollout_indices, 10:13] - self.last_root_vel[rollout_indices, 3:]
        ) / self.dt
        self.projected_gravity[rollout_indices] = quat_rotate_inverse(
            self.base_quat[rollout_indices],
            self.gravity_vec[rollout_indices]
        )

        # Update foot positions and velocities for rollout environments
        # FIXME: Should not update main_env
        self.foot_positions = self.rigid_body_state.view(self.total_num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.foot_velocities = self.rigid_body_state.view(self.total_num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self._post_physics_step_callback_rollout()

        # Compute observations and rewards for rollout environments
        # We don't need to check for termination since rollout environments
        # can only be reset when their main environment is reset
        # FIXME: this might distrub main env reward computation
        self.compute_reward_rollout()
        self.compute_observations()

        # Update last actions, velocities, and root velocities for rollout environments
        self.last_actions[rollout_indices] = self.actions[rollout_indices].clone()
        self.last_dof_vel[rollout_indices] = self.dof_vel[rollout_indices].clone()
        self.last_root_vel[rollout_indices] = self.root_states[rollout_indices, 7:13].clone()

    def _post_physics_step_callback(self):
        """ Callback after physics step. Override as needed. """
        # Update commands based on resampling time
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)
                   == 0).nonzero(as_tuple=False).flatten()

        # Filter to only include main environments
        main_env_ids = env_ids[torch.isin(env_ids, self.main_env_indices)]
        if len(main_env_ids) > 0:
            self._resample_commands(main_env_ids)

            # Propagate the commands from main to rollout envs
            for i in range(len(main_env_ids)):
                main_idx = main_env_ids[i]
                rollout_indices = torch.nonzero(self.rollout_to_main_map == main_idx).flatten()
                # Remove the main env itself from the list
                rollout_indices = rollout_indices[rollout_indices != main_idx]

                if len(rollout_indices) > 0:
                    self.commands[rollout_indices] = self.commands[main_idx].clone()

        if self.cfg.commands.heading_command:
            # Convert heading to ang vel command using P controller
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _post_physics_step_callback_rollout(self):
        pass

    # === Computings === #

    def check_termination(self):
        """ Check if environments need to be reset.
            Only main environments can terminate naturally.
            Rollout environments are reset when their main env is reset.
        """
        # Check termination for main environments
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf[self.main_env_indices] |= self.time_out_buf[self.main_env_indices]

        # Because rollout env will be synced with main env, we don't need to reset them
        # # For rollout environments, propagate reset from main environment
        # for i in range(self.num_main_envs):
        #     if self.reset_buf[self.main_env_indices[i]]:
        #         rollout_indices = self.main_to_rollout_indices[i]
        #         if len(rollout_indices) > 0:
        #             self.reset_buf[rollout_indices] = True

    def reset_idx(self, env_ids):
        """ Reset environments with specified IDs.
            For each main environment that's reset, also reset its rollout environments.
        """
        if len(env_ids) == 0:
            return

        # Update curriculum for main environments
        main_env_ids = env_ids[torch.isin(env_ids, self.main_env_indices)]
        if self.cfg.terrain.curriculum and len(main_env_ids) > 0:
            self._update_terrain_curriculum(main_env_ids)

        # Avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(main_env_ids)

        # Reset robot states for all specified environments
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # Resample commands for main environments and propagate them to rollout envs
        main_env_mask = torch.isin(env_ids, self.main_env_indices)
        main_envs_to_reset = env_ids[main_env_mask]

        if len(main_envs_to_reset) > 0:
            self._resample_commands(main_envs_to_reset)

            # Propagate the commands from main to rollout envs
            for i in range(len(main_envs_to_reset)):
                main_idx = main_envs_to_reset[i]
                rollout_indices = torch.nonzero(self.rollout_to_main_map == main_idx).flatten()
                # Remove the main env itself from the list
                rollout_indices = rollout_indices[rollout_indices != main_idx]

                if len(rollout_indices) > 0:
                    self.commands[rollout_indices] = self.commands[main_idx].clone()

        # Reset buffers for all specified environments
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.feet_contact_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # Fill extras for only the main environments
        if len(main_envs_to_reset) > 0:
            self.extras["episode"] = {}
            for key in self.episode_sums.keys():
                self.extras["episode"]['rew_' + key] = torch.mean(
                    self.episode_sums[key][main_envs_to_reset]
                ) / self.max_episode_length_s
                self.episode_sums[key][env_ids] = 0.

            # Log additional curriculum info
            if self.cfg.terrain.curriculum:
                self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            if self.cfg.commands.curriculum:
                self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            if self.cfg.rewards.multi_stage_rewards:
                self.extras["episode"]["reward_stage"] = float(self.reward_scales_stage)

            # Send timeout info to the algorithm
            if self.cfg.env.send_timeouts:
                self.extras["time_outs"] = self.time_out_buf

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.total_num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def compute_reward(self):
        """ Compute rewards for all environments.
            Rewards are only meaningful for main environments.
            For rollout environments, rewards are used for planning trajectory futures.
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)

        # Add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_reward_rollout(self):
        """ Compute rewards for all environments.
            Rewards are only meaningful for main environments.
            For rollout environments, rewards are used for planning trajectory futures.
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)

        # Add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew

    def compute_observations(self):
        """ Computes observations for all environments """
        # Compute base observations (same for all environments)
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 -
                                 self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # Make sure the noise_scale_vec has correct dimensions
        if self.add_noise:
            # Regenerate noise scale vector if necessary to match observation size
            if self.noise_scale_vec.shape[0] != self.obs_buf.shape[1]:
                self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

            # Only then add noise
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _compute_torques(self, actions, env_ids=None):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller,
            or directly as scaled torques.
        """
        # PD controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if control_type == "P":
            torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * \
                (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        ret = torch.clip(torques, -self.torque_limits, self.torque_limits)
        if env_ids is not None:
            return ret[env_ids]
        return ret

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw
        IMPORTANT: This method takes a lot of GPU memory.
        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.total_num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points),
                                    self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()  # convert float to indices
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.total_num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_env_origins(self):
        """ Sets environment origins with proper heights above terrain using raycast.
            For trimesh terrain, evaluates actual terrain height at each point.
            Otherwise creates a grid at nominal height.
            FIXME: logic to distinguish Curriculum/TerrainObj/ConfinedTerrainObj
        """
        self.nominal_height = self.cfg.rewards.base_height_target
        if self.cfg.terrain.mesh_type in ["trimesh", "confined_trimesh"]:
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

            if hasattr(self.cfg.terrain, 'random_origins') and self.cfg.terrain.random_origins:
                self.custom_origins = False # This is for curriculum terrain, not for TerrainObj
                # Random origin generation for multi-layer environments
                print("Using random origins generation for multi-layer environment")
                valid_origins_count = 0
                attempts = 0
                max_attempts = self.cfg.terrain.origin_generation_max_attempts
                required_clearance = self.nominal_height * self.cfg.terrain.height_clearance_factor

                # Arrays to store valid positions
                valid_positions = []

                # Define the range for random position sampling
                x_min, x_max = self.cfg.terrain.origins_x_range
                y_min, y_max = self.cfg.terrain.origins_y_range

                print(f"Generating random origins with clearance: {required_clearance:.2f} m")
                print(f"Position range X: [{x_min}, {x_max}], Y: [{y_min}, {y_max}]")

                # Store bounds for logging
                min_ground_height = float('inf')
                max_ground_height = float('-inf')
                min_ceiling_height = float('inf')
                max_ceiling_height = float('-inf')

                # Generate valid positions with sufficient height clearance
                while valid_origins_count < self.num_envs and attempts < max_attempts:
                    # Generate batch of random positions
                    batch_size = min(1000, max_attempts - attempts)
                    if batch_size <= 0:
                        break

                    # Generate random XY positions
                    rand_x = torch.rand(batch_size, device=self.device) * (x_max - x_min) + x_min
                    rand_y = torch.rand(batch_size, device=self.device) * (y_max - y_min) + y_min
                    positions = torch.stack([rand_x, rand_y], dim=1)

                    # Update attempt counter
                    attempts += batch_size

                    if hasattr(self, 'terrain') and hasattr(self.terrain, 'get_heights_batch'):
                        try:
                            # Get ground heights by casting rays downward
                            ground_heights = self.terrain.get_heights_batch(positions.cpu(), max_height=20.0, cast_dir=1)
                            ground_heights = torch.from_numpy(ground_heights).to(self.device)

                            # Get ceiling heights by casting rays upward
                            ceiling_heights = self.terrain.get_heights_batch(positions.cpu(), max_height=20.0, cast_dir=-1)
                            ceiling_heights = torch.from_numpy(ceiling_heights).to(self.device)

                            # Calculate clearance between ceiling and ground
                            clearance = ceiling_heights - ground_heights

                            # Update min/max bounds for logging
                            min_ground_height = min(min_ground_height, ground_heights.min().item())
                            max_ground_height = max(max_ground_height, ground_heights.max().item())
                            if ceiling_heights.numel() > 0:
                                min_ceiling_height = min(min_ceiling_height, ceiling_heights.min().item())
                                max_ceiling_height = max(max_ceiling_height, ceiling_heights.max().item())

                            # Find positions with sufficient clearance
                            # valid_indices = (clearance > required_clearance).nonzero(as_tuple=True)[0]
                            valid1 = clearance > required_clearance
                            valid2 = clearance < 1e-6  # Only one layer
                            valid_indices = torch.where(valid1 | valid2)[0]

                            if len(valid_indices) > 0:
                                for idx in valid_indices:
                                    if valid_origins_count < self.num_envs:
                                        # Store valid position with ground height
                                        valid_positions.append((
                                            positions[idx, 0].item(),
                                            positions[idx, 1].item(),
                                            ground_heights[idx].item()
                                        ))
                                        valid_origins_count += 1
                                    else:
                                        break
                        except Exception as e:
                            print(f"Error getting terrain heights: {e}")
                            import traceback
                            traceback.print_exc()

                    # Print progress occasionally
                    if attempts % 5000 == 0 or valid_origins_count >= self.num_envs:
                        print(
                            f"Origin generation: {valid_origins_count}/{self.num_envs} valid positions found after {attempts} attempts")
                        print(f"Height ranges - Ground: [{min_ground_height:.2f}, {max_ground_height:.2f}], "
                              f"Ceiling: [{min_ceiling_height:.2f}, {max_ceiling_height:.2f}]")

                if valid_origins_count < self.num_envs:
                    print(
                        f"Warning: Could only find {valid_origins_count}/{self.num_envs} valid positions after {attempts} attempts")
                    print("Some environments may be placed at fallback positions")

                    # Fill any remaining positions with valid ones (repeating if necessary)
                    if valid_origins_count > 0:
                        for i in range(valid_origins_count, self.num_envs):
                            idx = i % valid_origins_count  # Wrap around to reuse valid positions
                            valid_positions.append(valid_positions[idx])
                    else:
                        # If no valid positions found, use a grid as fallback
                        print("No valid positions found, falling back to grid layout")
                        for i in range(self.num_envs):
                            valid_positions.append((
                                (i % 10) * 3.0,  # Simple grid layout
                                (i // 10) * 3.0,
                                self.nominal_height
                            ))

                # Set the environment origins from valid positions
                for i in range(self.num_envs):
                    self.env_origins[i, 0] = valid_positions[i][0]
                    self.env_origins[i, 1] = valid_positions[i][1]
                    self.env_origins[i, 2] = valid_positions[i][2]

                # Log position range of final origins
                print(f"Final environment positions range:")
                print(f"X: [{self.env_origins[:, 0].min().item():.2f}, {self.env_origins[:, 0].max().item():.2f}]")
                print(f"Y: [{self.env_origins[:, 1].min().item():.2f}, {self.env_origins[:, 1].max().item():.2f}]")
                print(f"Z: [{self.env_origins[:, 2].min().item():.2f}, {self.env_origins[:, 2].max().item():.2f}]")

            else:
                self.custom_origins = True
                self.env_origins = torch.zeros(self.total_num_envs, 3, device=self.device, requires_grad=False)
                # Set random orientations for the created environments
                self.terrain_levels = torch.randint(
                    0, min(self.cfg.terrain.max_init_terrain_level + 1, self.cfg.terrain.num_rows), 
                    (self.total_num_envs,), device=self.device
                )
                self.terrain_types = torch.randint(
                    0, self.cfg.terrain.num_cols, (self.total_num_envs,), device=self.device
                )
                self.max_terrain_level = self.cfg.terrain.num_rows
                self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

                # For all main environments
                for i in range(self.num_main_envs):
                    main_idx = self.main_env_indices[i]
                    rollout_indices = self.main_to_rollout_indices[i]

                    # Set terrain level and type for main env
                    self.env_origins[main_idx] = self.terrain_origins[
                        self.terrain_levels[main_idx], self.terrain_types[main_idx]
                    ]

                    # Set the same terrain level and type for its rollout envs
                    if len(rollout_indices) > 0:
                        self.terrain_levels[rollout_indices] = self.terrain_levels[main_idx].clone()
                        self.terrain_types[rollout_indices] = self.terrain_types[main_idx].clone()
                        # Set the same environment origin for rollout envs as their main env
                        self.env_origins[rollout_indices] = self.env_origins[main_idx].clone()

        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

            # Create a grid of robots at nominal height with centering
            num_cols = int(np.floor(np.sqrt(self.num_envs)))
            num_rows = int(np.ceil(self.num_envs / num_cols))
            spacing = self.cfg.env.env_spacing

            # Generate centralized grid coordinates
            row_indices = torch.arange(num_rows, device=self.device)
            col_indices = torch.arange(num_cols, device=self.device)

            # Calculate offsets to center the grid
            row_offset = (num_rows - 1) / 2
            col_offset = (num_cols - 1) / 2

            # Apply offsets to create centered grid
            centered_rows = (row_indices - row_offset) * spacing
            centered_cols = (col_indices - col_offset) * spacing

            # Create grid using meshgrid
            xx, yy = torch.meshgrid(centered_rows, centered_cols)

            # Assign grid coordinates
            self.env_origins[:, 0] = xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.0

    # === Gets === #

    def get_observations(self):
        """ Main observations for the main environments """
        return self.obs_buf[self.main_env_indices]

    def get_observations_rollout(self):
        """ Rollout observations for the rollout environments """
        return self.obs_buf[self.rollout_env_indices]

    def get_observations_all(self):
        """ All observations for all environments """
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf[self.main_env_indices] if self.privileged_obs_buf is not None else None

    # === Resets & Updates === #

    def _resample_commands(self, env_ids):
        """ Randomly select commands for specified environments.
            For batch rollout, this is typically only done for main environments.
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1],
                (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1), device=self.device).squeeze(1)

        # Set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def set_commands(self, env_ids, commands):
        """ Set specific commands for selected environments.

        Args:
            env_ids (torch.Tensor): Tensor of environment indices to set commands for.
            commands (torch.Tensor): Tensor of commands to set, shape (len(env_ids), num_commands).
        """
        if commands.shape[0] != len(env_ids) or commands.shape[1] != self.commands.shape[1]:
            raise ValueError(f"Expected commands shape ({len(env_ids)}, {self.commands.shape[1]}), got {commands.shape}")

        self.commands[env_ids] = commands

    def set_all_commands(self, commands):
        """ Set specific commands for all environments.

        Args:
            commands (torch.Tensor): Tensor of commands to set, shape (num_envs, num_commands).
        """
        if commands.shape[0] != self.total_num_envs or commands.shape[1] != self.commands.shape[1]:
            raise ValueError(f"Expected commands shape ({self.num_envs}, {self.commands.shape[1]}), got {commands.shape}")

        self.commands[:] = commands

    def _reset_dofs(self, env_ids):
        """ Reset DOF position and velocities of selected environments """
        self.dof_pos[env_ids] = self.default_dof_pos * \
            torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids):
        """Reset ROOT states position and velocities of selected environments"""
        # Base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

            # XY position within 1.2m of the center with random offsets
            xy_offset = torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
            self.root_states[env_ids, :2] += xy_offset

            # For terrain environments, adjust Z position based on terrain height at the new position
            if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
                # Sample terrain heights at the new XY positions
                points = self.root_states[env_ids, :2].clone().unsqueeze(1)  # Get XY positions
                points += self.terrain.cfg.border_size
                points = (points/self.terrain.cfg.horizontal_scale).long()  # Convert to indices
                px = torch.clip(points[:, :, 0].view(-1), 0, self.height_samples.shape[0]-2)
                py = torch.clip(points[:, :, 1].view(-1), 0, self.height_samples.shape[1]-2)

                # Get height at the exact position
                heights = self.height_samples[px, py] * self.terrain.cfg.vertical_scale

                # Apply the correct height to the root state
                self.root_states[env_ids, 2] = heights + self.base_init_state[2]
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # Base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def _push_robots(self):
        """ Randomly pushes the robots. Emulates an impulse by setting a randomized base velocity. """
        # Only push main environments
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[self.main_env_indices, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_main_envs, 2), device=self.device
        )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum for terrain difficulty """
        if not self.init_done:
            # Don't change on initial reset
            return

        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # Robots that walked far enough progress to harder terrains
        move_up = distance > self.terrain.env_length / 2
        # Robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~move_up

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0)
        )
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum
            )

    def _sync_main_to_rollout(self):
        """ Synchronize states from main environments to their corresponding rollout environments.
            This ensures rollout environments start from the same state as their parent main environment.
            Optimized for better performance by using batched operations.
        """
        if len(self.rollout_env_indices) == 0:
            return  # No rollout environments to sync
        # Prepare mapping tensor for efficient batch operation
        # For each rollout env, stores which main env to copy from
        source_indices = torch.zeros(self.total_num_envs, dtype=torch.long, device=self.device)

        # Only process rollout environments
        for i in range(self.num_main_envs):
            main_idx = self.main_env_indices[i]
            rollout_indices = self.main_to_rollout_indices[i]

            if len(rollout_indices) > 0:
                # Set source index for all rollout environments of this main env
                source_indices[rollout_indices] = main_idx

        # Get only rollout environments for efficient indexing
        rollout_env_mask = torch.zeros(self.total_num_envs, dtype=torch.bool, device=self.device)
        rollout_env_mask[self.rollout_env_indices] = True
        rollout_sources = source_indices[rollout_env_mask]

        # Use vectorized operations to copy all states at once
        # Root state
        self.root_states[self.rollout_env_indices] = self.root_states[rollout_sources]

        # DOF state
        self.dof_pos[self.rollout_env_indices] = self.dof_pos[rollout_sources]
        self.dof_vel[self.rollout_env_indices] = self.dof_vel[rollout_sources]

        # Actions and history
        self.actions[self.rollout_env_indices] = self.actions[rollout_sources]
        self.last_actions[self.rollout_env_indices] = self.last_actions[rollout_sources]
        self.last_dof_vel[self.rollout_env_indices] = self.last_dof_vel[rollout_sources]
        self.last_root_vel[self.rollout_env_indices] = self.last_root_vel[rollout_sources]

        # Base state
        self.base_pos[self.rollout_env_indices] = self.base_pos[rollout_sources]
        self.base_quat[self.rollout_env_indices] = self.base_quat[rollout_sources]
        self.base_lin_vel[self.rollout_env_indices] = self.base_lin_vel[rollout_sources]
        self.base_ang_vel[self.rollout_env_indices] = self.base_ang_vel[rollout_sources]
        self.projected_gravity[self.rollout_env_indices] = self.projected_gravity[rollout_sources]

        if self.cfg.domain_rand.rollout_envs_sync_pos_drift > 0.:
            # Apply random drift to the root position of rollout environments
            # IMPORTANT: This may avoid sim slow down (foundLostAggregatePairsCapacity)
            self.base_pos[self.rollout_env_indices] += (torch.rand_like(self.base_pos[self.rollout_env_indices])-0.5)\
                                                        * self.cfg.domain_rand.rollout_envs_sync_pos_drift

        # Feet and contact state
        self.feet_air_time[self.rollout_env_indices] = self.feet_air_time[rollout_sources]
        self.feet_contact_time[self.rollout_env_indices] = self.feet_contact_time[rollout_sources]
        self.last_contacts[self.rollout_env_indices] = self.last_contacts[rollout_sources]

        # # BUG: this approach may lead to reset failure (unknown reason)
        # # Apply state changes to simulator in a single batched operation
        # env_ids_int32 = self.rollout_env_indices.to(dtype=torch.int32)

        # # Update DOF state
        # self.gym.set_dof_state_tensor_indexed(
        #     self.sim,
        #     gymtorch.unwrap_tensor(self.dof_state),
        #     gymtorch.unwrap_tensor(env_ids_int32),
        #     len(env_ids_int32)
        # )

        # # Update root state
        # self.gym.set_actor_root_state_tensor_indexed(
        #     self.sim,
        #     gymtorch.unwrap_tensor(self.root_states),
        #     gymtorch.unwrap_tensor(env_ids_int32),
        #     len(env_ids_int32)
        # )

        # Temprarily update whole state tensor
        self.gym.set_dof_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
        )

        self.gym.set_actor_root_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
        )

        self.t_rollout = self.t_main

    def _cache_main_env_states(self):
        """Cache the current state of main environments to restore after rollouts.
        This allows us to freeze main environments while rollout environments explore.
        """
        # Get main environment indices
        main_indices = self.main_env_indices

        # Create cache for main environments' state if it doesn't exist
        if not hasattr(self, 'main_env_cache'):
            self.main_env_cache = {
                'root_states': torch.zeros_like(self.root_states[main_indices]),
                'dof_pos': torch.zeros_like(self.dof_pos[main_indices]),
                'dof_vel': torch.zeros_like(self.dof_vel[main_indices]),
                'actions': torch.zeros_like(self.actions[main_indices]),
                'last_actions': torch.zeros_like(self.last_actions[main_indices]),
                'last_dof_vel': torch.zeros_like(self.last_dof_vel[main_indices]),
                'last_root_vel': torch.zeros_like(self.last_root_vel[main_indices]),
                'base_pos': torch.zeros_like(self.base_pos[main_indices]),
                'base_quat': torch.zeros_like(self.base_quat[main_indices]),
                'base_lin_vel': torch.zeros_like(self.base_lin_vel[main_indices]),
                'base_ang_vel': torch.zeros_like(self.base_ang_vel[main_indices]),
                'base_lin_acc': torch.zeros_like(self.base_lin_acc[main_indices]),
                'base_ang_acc': torch.zeros_like(self.base_ang_acc[main_indices]),
                'projected_gravity': torch.zeros_like(self.projected_gravity[main_indices]),
                'feet_air_time': torch.zeros_like(self.feet_air_time[main_indices]),
                'feet_contact_time': torch.zeros_like(self.feet_contact_time[main_indices]),
                'last_contacts': torch.zeros_like(self.last_contacts[main_indices]),
            }

        # Cache the current state of main environments
        self.main_env_cache['root_states'] = self.root_states[main_indices].clone()
        self.main_env_cache['dof_pos'] = self.dof_pos[main_indices].clone()
        self.main_env_cache['dof_vel'] = self.dof_vel[main_indices].clone()
        self.main_env_cache['actions'] = self.actions[main_indices].clone()
        self.main_env_cache['last_actions'] = self.last_actions[main_indices].clone()
        self.main_env_cache['last_dof_vel'] = self.last_dof_vel[main_indices].clone()
        self.main_env_cache['last_root_vel'] = self.last_root_vel[main_indices].clone()
        self.main_env_cache['base_pos'] = self.base_pos[main_indices].clone()
        self.main_env_cache['base_quat'] = self.base_quat[main_indices].clone()
        self.main_env_cache['base_lin_vel'] = self.base_lin_vel[main_indices].clone()
        self.main_env_cache['base_ang_vel'] = self.base_ang_vel[main_indices].clone()
        self.main_env_cache['base_lin_acc'] = self.base_lin_acc[main_indices].clone()
        self.main_env_cache['base_ang_acc'] = self.base_ang_acc[main_indices].clone()
        self.main_env_cache['projected_gravity'] = self.projected_gravity[main_indices].clone()
        self.main_env_cache['feet_air_time'] = self.feet_air_time[main_indices].clone()
        self.main_env_cache['feet_contact_time'] = self.feet_contact_time[main_indices].clone()
        self.main_env_cache['last_contacts'] = self.last_contacts[main_indices].clone()

    def _restore_main_env_states(self):
        """Restore the cached state of main environments after rollouts."""
        if not hasattr(self, 'main_env_cache'):
            print("Warning: Attempted to restore main environment states without cache.")
            return

        # Get main environment indices
        main_indices = self.main_env_indices

        # Restore the state of main environments from cache
        self.root_states[main_indices] = self.main_env_cache['root_states'].clone()
        self.dof_pos[main_indices] = self.main_env_cache['dof_pos'].clone()
        self.dof_vel[main_indices] = self.main_env_cache['dof_vel'].clone()
        self.actions[main_indices] = self.main_env_cache['actions'].clone()
        self.last_actions[main_indices] = self.main_env_cache['last_actions'].clone()
        self.last_dof_vel[main_indices] = self.main_env_cache['last_dof_vel'].clone()
        self.last_root_vel[main_indices] = self.main_env_cache['last_root_vel'].clone()
        self.base_pos[main_indices] = self.main_env_cache['base_pos'].clone()
        self.base_quat[main_indices] = self.main_env_cache['base_quat'].clone()
        self.base_lin_vel[main_indices] = self.main_env_cache['base_lin_vel'].clone()
        self.base_ang_vel[main_indices] = self.main_env_cache['base_ang_vel'].clone()
        self.base_lin_acc[main_indices] = self.main_env_cache['base_lin_acc'].clone()
        self.base_ang_acc[main_indices] = self.main_env_cache['base_ang_acc'].clone()
        self.projected_gravity[main_indices] = self.main_env_cache['projected_gravity'].clone()
        self.feet_air_time[main_indices] = self.main_env_cache['feet_air_time'].clone()
        self.feet_contact_time[main_indices] = self.main_env_cache['feet_contact_time'].clone()
        self.last_contacts[main_indices] = self.main_env_cache['last_contacts'].clone()

        # # Apply restored states to the simulation
        # env_ids_int32 = main_indices.to(dtype=torch.int32)

        # # Update DOF state in the simulator
        # self.gym.set_dof_state_tensor_indexed(
        #     self.sim,
        #     gymtorch.unwrap_tensor(self.dof_state),
        #     gymtorch.unwrap_tensor(env_ids_int32),
        #     len(env_ids_int32)
        # )

        # # Update root state in the simulator
        # self.gym.set_actor_root_state_tensor_indexed(
        #     self.sim,
        #     gymtorch.unwrap_tensor(self.root_states),
        #     gymtorch.unwrap_tensor(env_ids_int32),
        #     len(env_ids_int32)
        # )

        # Temporarily update whole state tensor
        self.gym.set_dof_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
        )
        self.gym.set_actor_root_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
        )

    # === Configs === #

    def _parse_cfg(self, cfg):
        """ Parse configuration to update internal variables """
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.max_episode_length_s = self.max_episode_length * self.dt

        self.cfg.domain_rand.push_interval = int(self.cfg.domain_rand.push_interval_s / self.dt)

        # Debug visualization flags
        if hasattr(self.cfg.viewer, 'debug_viz_origins'):
            self.debug_viz_origins = self.cfg.viewer.debug_viz_origins

        # Reward scales
        self.reward_scales = self._get_reward_scales()
        # Set to min stage of rewards
        self.reward_scales_stage = self.cfg.rewards.reward_min_stage

        # Command ranges
        self.command_ranges = {
            "lin_vel_x": self.cfg.commands.ranges.lin_vel_x,
            "lin_vel_y": self.cfg.commands.ranges.lin_vel_y,
        }

        if self.cfg.commands.heading_command:
            self.command_ranges["heading"] = self.cfg.commands.ranges.heading
        else:
            self.command_ranges["ang_vel_yaw"] = self.cfg.commands.ranges.ang_vel_yaw

        # Load obs scales
        self.obs_scales = self.cfg.normalization.obs_scales

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Removes zero scales + multiplies non-zero ones by dt
        """
        # Remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        # Prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # Reward episode sums
        self.episode_sums = {
            name: torch.zeros(
                self.total_num_envs, dtype=torch.float, device=self.device, requires_grad=False
            ) for name in self.reward_scales.keys()
        }

    def _get_reward_scales(self, stage=0):
        self.reward_scales_dict = class_to_dict(self.cfg.rewards.scales)
        if self.cfg.rewards.multi_stage_rewards:
            reward_scales = {}
            for key, value in self.reward_scales_dict.items():
                if not isinstance(value, list):
                    reward_scales[key] = value
                else:
                    if stage >= len(value):
                        reward_scales[key] = value[-1]
                    else:
                        reward_scales[key] = value[stage]
            return reward_scales
        else:
            return self.reward_scales_dict

    def update_reward_scales(self, mean_reward):
        if mean_reward > self.cfg.rewards.reward_stage_threshold and \
                self.reward_scales_stage < self.cfg.rewards.reward_max_stage:
            self.reward_scales_stage += 1
            self.reward_scales = self._get_reward_scales(self.reward_scales_stage)
            self._prepare_reward_function()
            return True
        return False

    # === Utils === #

    def set_camera(self, position, lookat):
        """ Set camera position and direction """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """

        # draw height lines
        self.gym.clear_lines(self.viewer)
        
        # Draw environment origins as orange spheres with radius 0.5m
        if self.debug_viz_origins:
            sphere_geom = gymutil.WireframeSphereGeometry(0.2, 16, 16, None, color=(1, 0.5, 0))
            for j in range(self.num_envs):
                i = self.main_env_indices[j]
                origin_pos = self.env_origins[i].cpu().numpy()
                sphere_pose = gymapi.Transform(gymapi.Vec3(origin_pos[0], origin_pos[1], origin_pos[2]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        
        # Draw height lines if terrain with height measurements exists
        if "terrain" in self.__dir__() and self.terrain.cfg.measure_heights:
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            height_sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
            for j in range(self.num_envs):
                i = self.main_env_indices[j]
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(height_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
