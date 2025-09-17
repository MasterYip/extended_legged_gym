import os
import torch
import numpy as np
from isaacgym import gymapi
from isaacgym.torch_utils import *
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_pose_adapt import BasePoseAdapt
from legged_gym.envs.base.base_pose_adapt_config import BasePoseAdaptCfg
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.envs.base.base_pose_adapt_config import BasePoseAdaptCfgPPO


class AnymalCBasePoseCtrlCfg(BasePoseAdaptCfg):
    """Configuration for ANYmal C base pose control task.
    
    This task directly controls the robot's base pose with PD control,
    without using a neural network. It's designed to test the PD control 
    performance and tune gains.
    """
    # Randomization options
    randomize_init_pos: bool = False  # Randomize initial position for testing 
    randomize_init_yaw: bool = False  # Randomize initial yaw for testing
    
    class env:
        episode_length_s: float = 20  # Longer episodes for testing
        num_envs: int = 64  # Fewer environments for debugging
        num_observations: int = 132  # Will be updated dynamically based on actual obs
        num_privileged_obs: int = None
        num_actions: int = 6  # linear vel (3) + angular vel (3)
        env_spacing = 3.0
    class asset(BasePoseAdaptCfg.asset):
        file: str = "{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_c_base.urdf"
        name: str = "anymal_c_base"
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

    class init_state:
        # Initial pose for the robot
        pos = [0.0, 0.0, 2]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {}  # empty for base-only robot

    class commands:
        # Commands for direct control
        num_commands: int = 3  # lin_x, lin_y, ang_yaw
        resampling_time: float = 10.0  # Longer time for testing patterns
        lin_vel_x: list = [-0.0, 0.0]  # Limited velocity range for testing
        lin_vel_y: list = [-0.0, 0.0]  # Limited velocity range for testing
        ang_vel_yaw: list = [-0.0, 0.0]  # Limited angular velocity for testing
        heading_command: bool = False  # Use direct velocity commands for testing

    class control(BasePoseAdaptCfg.control):
        decimation: int = 2  # Control frequency decimation
        # PD control parameters for base pose adjustment - tuned for better tracking
        position_p_gain: float = 90000.0  # Higher P gain for position
        position_d_gain: float = 2000.0    # Higher D gain for damping
        rotation_p_gain: float = 500.0   # Higher P gain for rotation
        rotation_d_gain: float = 40.0    # Higher D gain for angular damping

    class raycaster(BasePoseAdaptCfg.raycaster):
        # Ray casting parameters - simple setup for debugging
        enable_raycast: bool = True  # Disable raycasting for basic PD testing
        terrain_file: str = "/home/user/CodeSpace/Python/terrains/terrain.obj"
        ray_pattern: str = "spherical"
        spherical_num_azimuth: int = 8
        spherical_num_elevation: int = 4
        max_distance: float = 2.0
        
        # Visualization options
        draw_rays: bool = False
        draw_mesh: bool = False
        draw_hits: bool = False

    class rewards(BasePoseAdaptCfg.rewards):
        # Reward weights - modified for testing PD control
        collision_penalty: float = 0.0  # No collision penalty for testing
        close_distance_penalty: float = 0.0  # No distance penalty for testing
        nominal_alignment_reward: float = 1.0  # Focus on alignment with nominal pose
        max_contact_force: float = 100.0  # Higher threshold to avoid early termination
        min_safe_distance: float = 0.1

        # Command tracking rewards
        lin_vel_tracking: float = 1.0  # Higher weights for velocity tracking
        ang_vel_tracking: float = 1.0

    class obstacles:
        enable: bool = False  # No obstacles for basic testing


class AnymalCBasePoseCtrlCfgPPO(BasePoseAdaptCfgPPO):
    """PPO configuration parameters specifically for AnymalC base pose control task."""
    
    seed: int = 1
    runner_class_name = "OnPolicyRunner"
    
    class algorithm(BasePoseAdaptCfgPPO.algorithm):
        # Additional/overridden PPO algorithm parameters for AnymalC
        entropy_coef: float = 0.005  # Lower entropy for more focused exploration
        num_learning_epochs: int = 8  # More epochs for better convergence with complex terrain

    class runner(BasePoseAdaptCfgPPO.runner):
        # Runner parameters specific to AnymalC
        max_iterations: int = 2000  # More iterations for this complex task
        save_interval: int = 50  # Save model interval (in iterations)
        experiment_name: str = "anymal_c_base_pose_ctrl"
        run_name: str = ""
        multi_stage_rewards: bool = False  # Enable/disable multi-stage rewards
        
    class policy(BasePoseAdaptCfgPPO.policy):
        # Policy architecture parameters
        init_noise_std: float = 0.8  # Initial action noise std
        actor_hidden_dims: list = [256, 128, 64]  # Larger network for complex task
        critic_hidden_dims: list = [256, 128, 64]  # Larger network for complex task
        activation: str = "elu"  # Hidden layer activation


class AnymalCBasePoseCtrl(BasePoseAdapt):
    """ANYmal C base pose control task.
    
    This task directly uses the command input to control the robot's base pose,
    bypassing the neural network for testing PD control performance.
    """
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Set Anymal C specific parameters
        self.nominal_height = 0.5  # Appropriate height for Anymal C
        self.base_index = 0  # Will be properly set in _create_envs
        
        # Initialize the base class
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # For testing different command patterns
        self.test_pattern_enabled = True
        self.test_pattern_timer = torch.zeros(self.num_envs, device=self.device)
        self.test_patterns = [
            # [lin_x, lin_y, ang_yaw]
            [0.5, 0.0, 0.0],    # Forward
            [0.0, 0.5, 0.0],    # Sideways
            [0.0, 0.0, 0.3],    # Rotate
            [0.5, 0.5, 0.0],    # Diagonal
            [0.5, 0.0, 0.3],    # Forward + rotate
            [0.0, 0.0, 0.0],    # Stop
        ]
        self.current_pattern_idx = 0
        self.pattern_duration = 1.0  # seconds
        
    def step(self, actions):
        """Override step to directly use commands as actions.
        
        This bypasses the policy and allows direct testing of PD control.
        """
        # Update test pattern if enabled
        if self.test_pattern_enabled:
            self._update_test_pattern()
            
        # Use commands directly as actions instead of policy output
        commands = torch.zeros(self.num_envs, 6, device=self.device)
        commands[:, :2] = self.commands[:, :2]
        commands[:, 5] = self.commands[:, 2]
        return super().step(commands)
    
    def _update_test_pattern(self):
        """Update test pattern for command testing."""
        self.test_pattern_timer += self.dt
        change_pattern = self.test_pattern_timer >= self.pattern_duration
        
        if change_pattern.any():
            # Reset timer for environments that need pattern change
            self.test_pattern_timer[change_pattern] = 0
            
            # Move to next pattern
            self.current_pattern_idx = (self.current_pattern_idx + 1) % len(self.test_patterns)
            pattern = self.test_patterns[self.current_pattern_idx]
            
            # Set commands for all environments
            self.commands[:, 0] = pattern[0]  # lin_x
            self.commands[:, 1] = pattern[1]  # lin_y
            self.commands[:, 2] = pattern[2]  # ang_yaw
            
            print(f"Changed to pattern {self.current_pattern_idx}: {pattern}")

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _create_envs(self):
        """Create environments with Anymal C robots."""
        
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

        # save body names from the asset
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
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose,
                                                 self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            
    def _create_envs_AIgen(self):
        """Create environments with Anymal C robots."""
        
        # Get asset path
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # Setup asset options
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = True
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = False  # Allow the base to move
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.0

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

        # Load robot asset
        anymal_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Get number of bodies and DOFs
        self.num_bodies = self.gym.get_asset_rigid_body_count(anymal_asset)
        self.num_dofs = self.gym.get_asset_dof_count(anymal_asset)

        # Store body names and indices
        body_names = self.gym.get_asset_rigid_body_names(anymal_asset)
        self.dof_names = self.gym.get_asset_dof_names(anymal_asset)

        # Find base index
        self.base_index = self.gym.find_asset_rigid_body_index(anymal_asset, "base")

        # Calculate environment origins
        self._get_env_origins()
        
        # Create environments
        self.envs = []
        self.actor_handles = []

        # Define environment spacing
        spacing = 3.0  # Larger spacing for clearer visualization
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # Create environments with robots
        num_per_row = int(np.sqrt(self.num_envs))
        
        # Set initial pose from config
        start_pose = gymapi.Transform()
        # start_pose.p = gymapi.Vec3(*self.cfg.init_state.pos)
        # start_pose.r = gymapi.Quat(*self.cfg.init_state.rot)

        for i in range(self.num_envs):
            # Create the environment
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            
            # Set environment position based on origin
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            # Create the actor
            actor_handle = self.gym.create_actor(env_handle, anymal_asset, start_pose, self.cfg.asset.name, i, 1)

            # Disable DOF drive mode to allow PD control in simulation
            props = self.gym.get_actor_dof_properties(env_handle, actor_handle)
            for j in range(self.num_dofs):
                props["driveMode"][j] = gymapi.DOF_MODE_NONE
                props["stiffness"][j] = 0.0
                props["damping"][j] = 0.0

            self.gym.set_actor_dof_properties(env_handle, actor_handle, props)

            # Store the environment and actor handles
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            
    def reset_idx(self, env_ids):
        """Reset environments with the given IDs."""
        super().reset_idx(env_ids)
        
        # Reset test pattern timer for these environments
        if len(env_ids) > 0:
            self.test_pattern_timer[env_ids] = 0 

    def _draw_debug_vis(self):
        """Draw debug visualization for the robot."""
        # Clear previous visualizations
        if hasattr(self, 'vis'):
            self.vis.clear()
        else:
            self.gym.clear_lines(self.viewer)
        
        # Draw base velocity and command visualization
        cmd_vel_world = quat_rotate(self.target_quat, self.commands[:, :3]).cpu().numpy()
        cmd_vel_world[:, 2] = 0.0
        lin_vel = quat_rotate(self.base_quat, self.base_lin_vel).cpu().numpy()
        
        # For each environment, draw ray visualization
        if hasattr(self, 'ray_caster') and self.cfg.raycaster.enable_raycast:
            ray_data = self.ray_caster.data
            ray_dir_length = self.cfg.raycaster.max_distance  # Length of ray visualization
            
            # Draw ray visualization for each environment
            for i in range(self.num_envs):
                # Get base position
                base_pos = self.base_pos[i].cpu().numpy()

                # Draw target coordinate frame at robot position
                if hasattr(self, 'vis'):
                    # Draw base position and coordinate frame
                    quat_np = self.target_quat[i].cpu().numpy()
                    pos = self.target_pos[i].cpu()
                    self.vis.draw_frame_from_quat(
                        i, 
                        [quat_np[0], quat_np[1], quat_np[2], quat_np[3]],
                        pos,
                        width=0.02,
                        length=0.2
                    )
                    
                # Draw base velocity vector

                self.vis.draw_arrow(i, base_pos, base_pos + lin_vel[i], color=(0, 1, 0), width=0.02)
                
                # Draw target velocity vector
                self.vis.draw_arrow(i, base_pos, base_pos + cmd_vel_world[i], color=(1, 0, 0), width=0.02)
                
                # Draw target position
                self.vis.draw_point(i, self.target_pos[i], color=(1, 0, 0), size=0.05)
                
                
                # Draw rays if enabled
                if self.cfg.raycaster.draw_rays:
                    # Get ray directions and origins
                    ray_dir = self.ray_caster.ray_directions[i, :]
                    quat = self.base_quat[i].repeat(self.ray_caster.num_rays, 1)
                    
                    # Calculate ray origin position with offset
                    if hasattr(self.cfg.raycaster, 'offset_pos'):
                        offset = quat_rotate(quat[0:1], torch.tensor(
                            self.cfg.raycaster.offset_pos, device=self.device).unsqueeze(0)).squeeze(0).cpu().numpy()
                        ray_origin = base_pos + offset
                    else:
                        ray_origin = base_pos
                    
                    # Rotate ray directions to world frame
                    world_dir = quat_rotate(quat, ray_dir).cpu().numpy()
                    
                    # Draw each ray
                    for j in range(self.ray_caster.num_rays):
                        ray_end = ray_origin + world_dir[j] * ray_dir_length
                        
                        # Draw the ray using the visualizer if available
                        if hasattr(self, 'vis'):
                            self.vis.draw_line(i, [ray_origin, ray_end], color=(0.7, 0.7, 0.7))
                        else:
                            # Draw with basic line function if visualizer not available
                            self.gym.add_lines(
                                self.viewer, 
                                self.envs[i], 
                                1, 
                                [
                                    ray_origin[0], ray_origin[1], ray_origin[2],
                                    ray_end[0], ray_end[1], ray_end[2]
                                ], 
                                [0.7, 0.7, 0.7]
                            )
            
            # Draw hit points if enabled
            if self.cfg.raycaster.draw_hits:
                hit_mask = ray_data.ray_hits_found.view(self.num_envs, -1)
                
                # For each environment with hits
                for i in range(self.num_envs):
                    # Get hit points for this environment that actually had hits
                    env_hit_indices = hit_mask[i].nonzero(as_tuple=True)[0]
                    
                    if len(env_hit_indices) > 0:
                        env_hits = ray_data.ray_hits[i, env_hit_indices].cpu().numpy()
                        
                        # Draw each hit point
                        for hit_pos in env_hits:
                            if hasattr(self, 'vis'):
                                # Use visualizer to draw points
                                self.vis.draw_point(i, hit_pos, color=(1, 0, 0), size=0.05)

        return