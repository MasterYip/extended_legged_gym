from typing import Dict
from dataclasses import dataclass
import torch
from legged_gym.envs.base.base_config import BaseConfig
from isaacgym import gymapi


@dataclass
class BasePoseAdaptCfg(BaseConfig):

    class sim:
        dt: float = 0.005
        substeps: int = 1
        # gravity: list = [0.0, 0.0, -9.81]  # [m/s^2]
        gravity: list = [0.0, 0.0, 0.0]  # [m/s^2]
        up_axis: int = 1
        use_gpu_pipeline: bool = True
        physx: dict = None

        class physx:
            use_gpu: bool = True
            use_fabric: bool = False
            solver_type: int = 1  # 0: PGS, 1: TGS
            num_position_iterations: int = 4
            num_velocity_iterations: int = 0
            contact_offset: float = 0.01  # [m]
            rest_offset: float = 0.0  # [m]
            bounce_threshold_velocity: float = 0.5  # [m/s]
            max_depenetration_velocity: float = 1.0
            max_gpu_contact_pairs: int = 8388608  # 8*1024*1024
            num_threads: int = 4
            default_buffer_size_multiplier: int = 5
            contact_collection: int = 2  # 0: never, 1: last substep, 2: all substeps

    # Randomization options
    randomize_init_pos: bool = False  # Randomize initial position
    randomize_init_yaw: bool = False  # Randomize initial yaw

    # Not used
    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values
    # Not used

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class viewer:
        ref_env = 0
        pos = [10, 0, 0]  # [m]
        lookat = [0., 0., 0.]  # [m]

    class terrain:
        use_terrain_obj: bool = True  # use TerrainObj class to create terrain
        # path to the terrain file
        terrain_file: str = "/home/user/CodeSpace/Python/terrains/confined_terrain2.obj"
        mesh_type: str = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale: float = 0.1  # [m]
        vertical_scale: float = 0.005  # [m]
        border_size: float = 25.0  # [m]
        curriculum: bool = False
        static_friction: float = 1.0
        dynamic_friction: float = 1.0
        restitution: float = 0.0
        # Terrain dimensions
        terrain_length: float = 5.0
        terrain_width: float = 5.0
        num_rows: int = 8  # number of terrain rows (levels)
        num_cols: int = 8  # number of terrain cols (types)
        # Origin generation method
        random_origins: bool = True  # Use random origins instead of grid
        origin_generation_max_attempts: int = 10000  # Maximum attempts to generate valid random origins
        origins_x_range: list = [-20.0, 20.0]  # min/max range for random x position
        origins_y_range: list = [-20.0, 20.0]  # min/max range for random y position
        height_clearance_factor: float = 2.0  # Required clearance as multiple of nominal_height

    class commands:
        # Command configuration
        num_commands: int = 3  # lin_x, lin_y, ang_yaw
        resampling_time: float = 2.0  # Time in seconds before resampling commands
        lin_vel_x: list = [-0.5, 0.5]  # min max [m/s]
        lin_vel_y: list = [-0.5, 0.5]  # min max [m/s]
        ang_vel_yaw: list = [-0.5, 0.5]  # min max [rad/s]
        heading_command: bool = False  # Whether to include heading command

    class raycaster:
        # Ray casting parameters
        enable_raycast: bool = True
        ray_pattern: str = "spherical"  # Options: single, grid, cone, spherical
        spherical_num_azimuth: int = 16  # Number of rays in azimuth direction
        spherical_num_elevation: int = 8  # Number of rays in elevation direction
        num_rays: int = 32  # For cone pattern
        ray_angle: float = 60.0  # For cone pattern, angle in degrees
        max_distance: float = 2.0  # Maximum raycast distance
        terrain_file: str = ""  # Path to the terrain mesh file
        offset_pos: list = (0.0, 0.0, 0.0)  # Offset from the reference point
        attach_yaw_only: bool = False  # Whether to attach the yaw only

        # Visualization options
        draw_rays: bool = True  # Draw rays
        draw_mesh: bool = False  # Draw terrain mesh (expensive)
        draw_hits: bool = True  # Draw ray hit points

    class env:
        episode_length_s: float = 10  # Episode length in seconds
        num_envs: int = 4096
        num_observations: int = 150  # Will be updated dynamically based on actual obs
        num_privileged_obs: int = None
        num_actions: int = 6  # Linear and angular velocity
        env_spacing = 3.0

    class control:
        decimation: int = 5  # Control frequency decimation
        # PD control parameters for base pose adjustment
        position_p_gain: float = 50.0
        position_d_gain: float = 5.0
        rotation_p_gain: float = 50.0
        rotation_d_gain: float = 5.0
        action_scale: float = 1.0  # Scale factor for action inputs
        use_direct_pose_control: bool = False  # If True, uses direct pose setting instead of PD control

    class rewards:
        # Penalty weights
        collision_penalty: float = 1.0
        terrain_conformity_penalty: float = 1.0  # Penalty for base not conforming to terrain
        orientation_penalty: float = 0.2  # Small weight for penalizing non-flat orientation

        # Command tracking rewards
        lin_vel_tracking: float = 0.5
        ang_vel_tracking: float = 0.5

        # Downward velocity reward
        downward_vel_reward: float = 0.5  # Weight for the downward velocity reward
        downward_vel_scale: float = 0.5  # Scaling factor for the exponential reward function

        # Termination thresholds
        max_contact_force: float = 50.0  # N

    class normalization:
        clip_observations: float = 100.0
        clip_actions: float = 100.0

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = ["base"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # NOTE: It is said disable self collisions may save GPU memory.
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

        # Default nominal height for the robot base
        nominal_height: float = 0.25  # Default height for the robot base above ground [m]


    class init_state:
        pos = [0.0, 0.0, 1.]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "joint_a": 0.,
            "joint_b": 0.}


class BasePoseAdaptCfgPPO:
    """PPO configuration parameters for the BasePoseAdapt task."""

    seed: int = 1
    runner_class_name = "OnPolicyRunner"
    multi_stage_rewards: bool = False  # Enable/disable multi-stage rewards

    class algorithm:
        # PPO algorithm parameters
        value_loss_coef: float = 1.0
        use_clipped_value_loss: bool = True
        clip_param: float = 0.2
        entropy_coef: float = 0.01
        num_learning_epochs: int = 5
        num_mini_batches: int = 4  # Mini batches per epoch
        learning_rate: float = 1.0e-3  # Initial learning rate
        schedule: str = "adaptive"  # Learning rate schedule
        gamma: float = 0.99  # Discount factor
        lam: float = 0.95  # GAE lambda parameter
        desired_kl: float = 0.01  # Desired KL divergence for adaptive LR
        max_grad_norm: float = 1.0  # Gradient clipping

    class runner:
        # Runner parameters
        policy_class_name: str = "ActorCritic"
        algorithm_class_name: str = "PPO"
        num_steps_per_env: int = 24  # Steps per environment per iteration
        max_iterations: int = 1500  # Maximum number of training iterations
        save_interval: int = 50  # Save model interval (in iterations)
        experiment_name: str = "base_pose_adapt"
        run_name: str = ""
        # Load trained models
        resume: bool = False
        load_run: str = -1  # -1 = last run
        checkpoint: int = -1  # -1 = last saved model
        # Evaluation
        eval_freq: int = 50  # Evaluate model every N iterations
        num_eval_steps_per_env: int = 1000  # Steps per environment during evaluation

    class policy:
        # Policy architecture parameters
        init_noise_std: float = 1.0  # Initial action noise std
        actor_hidden_dims: list = [128, 64, 32]  # Actor network size
        critic_hidden_dims: list = [128, 64, 32]  # Critic network size
        activation: str = "elu"  # Hidden layer activation

    class normalization:
        # Observation normalization
        clip_observations: float = 10.0
        clip_actions: float = 1.0
