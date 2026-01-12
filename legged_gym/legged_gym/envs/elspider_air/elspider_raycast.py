'''
Author: Raymon Yip 2205929492@qq.com
Date: 2025-04-12 15:32:16
Description: ElSpider class with raycast perception
FilePath: /PredictiveDiffusionPlanner_Dev/legged_gym_cmp/legged_gym/legged_gym/envs/elspider_air/elspider_raycast.py
'''

from legged_gym.envs.base.legged_robot_raycast import LeggedRobotRayCast
from legged_gym.envs.base.legged_robot_depthcam import LeggedRobotDepth
from legged_gym.envs.elspider_air.elspider import ElSpider
from isaacgym import torch_utils
import torch
import numpy as np
from legged_gym.utils import GaitScheduler, GaitSchedulerCfg, AsyncGaitSchedulerCfg, AsyncGaitScheduler
from legged_gym.utils.helpers import class_to_dict
from isaacgym.torch_utils import quat_apply, quat_rotate, quat_rotate_inverse
from legged_gym import LEGGED_GYM_ROOT_DIR


class ElSpiderRayCast(LeggedRobotDepth):
    """ElSpider robot with raycast-based terrain perception.

    This class extends LeggedRobotRayCast to implement the ElSpider robot
    with raycast perception instead of height measurements.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # load actuator network
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)

        # Init gait scheduler
        cfg_gait = GaitSchedulerCfg()
        cfg_gait.dt = self.dt
        cfg_gait.period = 1.4
        cfg_gait.swing_height = 0.07
        self.gait_scheduler = GaitScheduler(self.height_samples,
                                            self.base_quat,
                                            self.base_lin_vel,
                                            self.base_ang_vel,
                                            self.projected_gravity,
                                            self.dof_pos,
                                            self.dof_vel,
                                            self.foot_positions,
                                            self.foot_velocities,
                                            self.num_envs,
                                            self.device,
                                            cfg_gait)

        # Initialize async gait scheduler
        cfg_async = AsyncGaitSchedulerCfg()
        self.async_gait_scheduler = AsyncGaitScheduler(self.height_samples,
                                                       self.base_quat,
                                                       self.base_lin_vel,
                                                       self.base_ang_vel,
                                                       self.projected_gravity,
                                                       self.dof_pos,
                                                       self.dof_vel,
                                                       self.foot_positions,
                                                       self.foot_velocities,
                                                       self.num_envs,
                                                       self.device,
                                                       cfg_async)

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        if hasattr(self.cfg.control, 'use_actuator_network') and self.cfg.control.use_actuator_network:
            self.sea_input = torch.zeros(self.num_envs*self.num_actions, 1, 2, device=self.device, requires_grad=False)
            self.sea_hidden_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
            self.sea_cell_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
            self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
            self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)

    def post_physics_step(self):
        """Update after physics step including gait schedulers."""
        super().post_physics_step()
        # Update gait scheduler
        self.gait_scheduler.step(self.foot_positions, self.foot_velocities, self.commands)

    def reset_idx(self, env_ids):
        """Reset environments with the given IDs."""
        # Use base class implementation for reset
        super().reset_idx(env_ids)

        # Reset ray caster for the specified environments
        if hasattr(self, 'ray_caster') and self.ray_caster is not None:
            self.ray_caster.reset(env_ids)

        # Additionaly empty actuator network hidden states
        if hasattr(self, 'sea_hidden_state_per_env'):
            self.sea_hidden_state_per_env[:, env_ids] = 0.
            self.sea_cell_state_per_env[:, env_ids] = 0.

    def _compute_torques(self, actions):
        # Choose between pd controller and actuator network
        if hasattr(self.cfg.control, 'use_actuator_network') and self.cfg.control.use_actuator_network:
            with torch.inference_mode():
                self.sea_input[:, 0, 0] = (actions * self.cfg.control.action_scale +
                                           self.default_dof_pos - self.dof_pos).flatten()
                self.sea_input[:, 0, 1] = self.dof_vel.flatten()
                torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(
                    self.sea_input, (self.sea_hidden_state, self.sea_cell_state))
            return torques
        else:
            # pd controller
            return super()._compute_torques(actions)

    def _reward_gait_scheduler(self):
        """Reward for tracking the gait scheduler."""
        return self.gait_scheduler.reward_foot_z_track()

    def _reward_async_gait_scheduler(self):
        """Reward for Async Gait Scheduler."""
        gait_scheduler_scales = class_to_dict(self.cfg.rewards.async_gait_scheduler)

        def get_weight(key, stage):
            if isinstance(gait_scheduler_scales[key], list):
                return gait_scheduler_scales[key][min(stage, len(gait_scheduler_scales[key])-1)]
            else:
                return gait_scheduler_scales[key]

        return self.async_gait_scheduler.reward_dof_align()*get_weight('dof_align', self.reward_scales_stage) + \
            self.async_gait_scheduler.reward_dof_nominal_pos()*get_weight('dof_nominal_pos', self.reward_scales_stage) + \
            self.async_gait_scheduler.reward_foot_z_align()*get_weight('reward_foot_z_align', self.reward_scales_stage)

    def _draw_debug_vis(self):
        """Draw debug visualization for the robot."""
        # Draw base velocity and command visualization
        self.gym.clear_lines(self.viewer)
        lin_vel = self.root_states[:, 7:10].cpu().numpy()
        cmd_vel_world = torch_utils.quat_rotate(self.base_quat, self.commands[:, :3]).cpu().numpy()
        cmd_vel_world[:, 2] = 0.0

        if 0:  # Terrain Mesh Debugging
            # Draw the WARP terrain mesh once - only for the first environment
            # This helps debug alignment between WARP and Isaac Gym terrain
            if hasattr(self, 'ray_caster') and not hasattr(self, '_mesh_drawn'):
                # Print TerrainObj mesh info if available
                if hasattr(self, 'terrain'):
                    from legged_gym.utils.terrain_obj import TerrainObj
                    if isinstance(self.terrain, TerrainObj):
                        # Get TerrainObj mesh information
                        terrain_bounds = self.terrain.terrain_mesh.bounds
                        terrain_center = (terrain_bounds[0] + terrain_bounds[1]) / 2
                        print(f"TerrainObj mesh bounds: {terrain_bounds}")
                        print(f"TerrainObj mesh center: {terrain_center}")
                        print(f"TerrainObj env origins: {self.terrain.env_origins[0, 0]}")

                        # Draw TerrainObj mesh center with a different color
                        self.vis.draw_point(0, terrain_center, color=(0, 1, 1), size=3)  # Cyan sphere

                # Print and visualize WARP mesh info
                for mesh_path, warp_mesh in self.ray_caster.meshes.items():
                    # Get mesh vertices from WARP
                    if hasattr(warp_mesh, 'points'):
                        # Convert WARP mesh vertices to numpy
                        vertices = warp_mesh.points.numpy()
                        indices = warp_mesh.indices.numpy().reshape(-1, 3)

                        # Calculate and print WARP mesh bounds and center
                        min_bounds = np.min(vertices, axis=0)
                        max_bounds = np.max(vertices, axis=0)
                        warp_center = (min_bounds + max_bounds) / 2
                        print(f"WARP mesh path: {mesh_path}")
                        print(f"WARP mesh bounds: [{min_bounds}, {max_bounds}]")
                        print(f"WARP mesh center: {warp_center}")

                        # Draw WARP mesh center with a different color
                        self.vis.draw_point(0, warp_center, color=(1, 0, 1), size=3)  # Magenta sphere

                        # Downsample for visualization (drawing all vertices/edges would be too slow)
                        step = max(1, len(vertices) // 10000)  # Limit to ~10k vertices for performance
                        vertices_subset = vertices[::step]

                        # Draw selected vertices as points
                        # self.vis.draw_points(0, vertices_subset, color=(0.8, 0.2, 0.8), size=0.02)

                        # Draw a subset of the edges to visualize the mesh structure
                        edge_step = max(1, len(indices) // 5000)  # Limit to ~5k edges for performance
                        for j in range(0, len(indices), edge_step):
                            face = indices[j]
                            # Draw the three edges of this triangle
                            for k in range(3):
                                v1 = vertices[face[k]]
                                v2 = vertices[face[(k+1) % 3]]
                                self.vis.draw_line(0, [v1, v2], color=(1.0, 0.5, 1.0))

                        print(f"Visualizing WARP mesh with {len(vertices_subset)} vertices and {len(indices)//edge_step} edges")

                # Mark mesh as drawn so we only do this expensive operation once
                # self._mesh_drawn = True

        for i in range(self.num_envs):
            base_pos = self.root_states[i, :3].cpu().numpy()
            # self.vis.draw_arrow(i, base_pos, base_pos + lin_vel[i], color=(0, 1, 0))
            # self.vis.draw_arrow(i, base_pos, base_pos + cmd_vel_world[i], color=(1, 0, 0))

            # Draw ray visualization if enabled
            ray_dir_length = 2
            if hasattr(self, 'ray_caster'):
                ray_data = self.ray_caster.data
                if 0:
                    # Get ray origins, directions and hits for this robot
                    ray_dir = self.ray_caster.ray_directions[i, :]
                    quat = self.root_states[i, 3:7].repeat(self.ray_caster.num_rays, 1)
                    ray_origins = base_pos + quat_rotate(quat[:1, :], torch.tensor(self.cfg.raycaster.offset_pos,
                                                                                device=self.device).unsqueeze(0)).cpu().numpy()
                    world_dir = quat_rotate(quat, ray_dir).cpu().numpy()
                    for j in range(self.ray_caster.num_rays):
                        # Draw ray direction (even if no hit)
                        ray_dir_end = ray_origins + world_dir[j] * ray_dir_length
                        self.vis.draw_line(i, [ray_origins, ray_dir_end], color=(0.7, 0.7, 0.7))  # Gray for direction

        ray_hits = ray_data.ray_hits.view(-1, 3).cpu().numpy()
        ray_hits = ray_hits[ray_data.ray_hits_found.view(-1).cpu().numpy() > 0]  # Filter out hits that are not found
        self.vis.draw_points(i, ray_hits, color=(1, 0, 0), size=0.02)

        return super()._draw_debug_vis()

    def draw_points(self, env_id, points, color=(1, 0, 0), size=0.02):
        """Draw 3D points in the environment.
        
        Args:
            env_id: Environment ID to draw in
            points: Array of 3D points to draw, shape (N, 3) or (3,)
            color: RGB color tuple (default red)
            size: Point size (default 0.02)
        """
        if not hasattr(self, 'vis') or self.vis is None:
            return
            
        # Convert points to numpy if it's a tensor
        if hasattr(points, 'cpu'):
            points = points.cpu().numpy()
        
        # Ensure points is 2D
        if len(points.shape) == 1:
            points = points.reshape(1, -1)
            
        # Draw the points
        self.vis.draw_points(env_id, points, color=color, size=size)
        # Draw head arrow
        base_pos = self.root_states[env_id, :3].cpu().numpy()
        forward = quat_apply(self.base_quat[env_id], self.forward_vec[env_id]).cpu().numpy()
        self.vis.draw_arrow(env_id, base_pos, base_pos + forward * 0.8, color=(0,1,0))

    def _reward_gait_2_step(self):
        # Foot index (alphabet): 0 LB, 1 LF, 2 LM, 3 RB, 4 RF, 5 RM
        # Hexapod 2-step gait: first group (0-1-5) synchronized, second group (2-3-4) synchronized
        # The two groups are asynchronized with each other
        
        # First group internal synchronization rewards (0-1-5)
        sync_lb_lf = self._sync_reward_func(0, 1)
        sync_lb_rm = self._sync_reward_func(0, 5)
        sync_lf_rm = self._sync_reward_func(1, 5)
        sync_group1 = (sync_lb_lf + sync_lb_rm + sync_lf_rm) / 3
        
        # Second group internal synchronization rewards (2-3-4)
        sync_lm_rb = self._sync_reward_func(2, 3)
        sync_lm_rf = self._sync_reward_func(2, 4)
        sync_rb_rf = self._sync_reward_func(3, 4)
        sync_group2 = (sync_lm_rb + sync_lm_rf + sync_rb_rf) / 3
        
        # Asynchronization rewards between the two groups
        async_lb_lm = self._async_reward_func(0, 2)
        async_lb_rb = self._async_reward_func(0, 3)
        async_lb_rf = self._async_reward_func(0, 4)
        async_lf_lm = self._async_reward_func(1, 2)
        async_lf_rb = self._async_reward_func(1, 3)
        async_lf_rf = self._async_reward_func(1, 4)
        async_rm_lm = self._async_reward_func(5, 2)
        async_rm_rb = self._async_reward_func(5, 3)
        async_rm_rf = self._async_reward_func(5, 4)
        
        # Calculate average asynchronization reward
        async_reward = (async_lb_lm + async_lb_rb + async_lb_rf + 
                         async_lf_lm + async_lf_rb + async_lf_rf + 
                         async_rm_lm + async_rm_rb + async_rm_rf) / 9
        
        # Calculate total synchronization reward
        sync_reward = (sync_group1 + sync_group2) / 2
        
        re = sync_reward + async_reward
        if self.cfg.commands.heading_command:
            re = re * torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > self.speed_min, 
                                    torch.abs(self.commands[:, 3]) >= self.speed_min/ 2)
        else:
            re = re * torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > self.speed_min, 
                                    torch.abs(self.commands[:, 2]) >= self.speed_min/ 2)
        return re
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        super().check_termination()
        
        # Add new termination condition - terminate if robot is upside down (z-component of projected gravity > 0)
        self.reset_buf |= (self.projected_gravity[:, 2] > 0) # Robot is upside down
        # self.reset_buf |= (self.base_lin_vel[:, 2] < -2.0)   # Fall off the terrain out of border
