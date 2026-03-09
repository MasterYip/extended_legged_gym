from legged_gym.envs.el_4090.el4090_spider_config import El4090SpiderCfg,El4090SpiderCfgPPO



class El4090SafeCfg(El4090SpiderCfg):
    class env(El4090SpiderCfg.env):
        num_envs = 10
        # 原始 66 维观测 + u_mu 77 维 = 143 维
        num_observations = 143
        # Debug settings
        debug_mode = False  # Enable debug output
        debug_interval = 100  # Print debug info every N steps
        debug_env_id = 0  # Which environment to debug (0-based index)

    class terrain(El4090SpiderCfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 10  # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1,
                             0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 0  # starting curriculum state
        terrain_length = 10.
        terrain_width = 10.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        difficulty_scale = 0.0
        terrain_proportions = [0., 1., 0., 0., 0.]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class asset(El4090SpiderCfg.asset):
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class control(El4090SpiderCfg.control):
        # PD Drive parameters matching Anymal:
        stiffness = {'HAA': 120., 
                     'HFE': 120., 
                     'KFE': 120.}  # [N*m/rad]
        damping = {'HAA': 1.2, 
                   'HFE': 1.2, 
                   'KFE': 1.2}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25  # Enable Network-0.5 | Disable Network-0.3

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"


    class asset(El4090SpiderCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/el_4090/urdf/el_4090.urdf"
        name = "el_4090"
        foot_name = "FOOT"
        collapse_fixed_joints = False # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        shoulder_name = "shoulder"
        penalize_contacts_on = ["BASE","SHANK","THIGH"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class init_state(El4090SpiderCfg.init_state):
        pos = [0.0, 0.0, 0.47]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "RF_HAA": 0.0,
            "RM_HAA": 0.0,
            "RB_HAA": 0.0,
            "LF_HAA": 0.0,
            "LM_HAA": 0.0,
            "LB_HAA": 0.0,

            "RF_HFE": 0.0,
            "RM_HFE": 0.0,
            "RB_HFE": 0.0,
            "LF_HFE": 0.0,
            "LM_HFE": 0.0,
            "LB_HFE": 0.0,

            "RF_KFE": 0.0,
            "RM_KFE": 0.0,
            "RB_KFE": 0.0,
            "LF_KFE": 0.0,
            "LM_KFE": 0.0,
            "LB_KFE": 0.0,
        }

    ## Rewards V1 (normal dof_acc)
    class rewards(El4090SpiderCfg.rewards):
        max_contact_force = 300.
        base_height_target = 0.5
        only_positive_rewards = False
        # Multi-stage
        # Stage 0: Learn to walk with tripod gait (with / w\o actuator net)
        # Stage 1: Correct DOF and FootZ positions / Prevent Slip
        multi_stage_rewards = True  # if true, reward scales should be list
        reward_stage_threshold = 5
        reward_min_stage = 0  # Start from 0
        reward_max_stage = 1

        class scales:

            termination = -0.0
            tracking_lin_vel = [6,5.5]
            tracking_ang_vel = [5.5,4.5]
            lin_vel_z = -5
            ang_vel_xy = -0.5
            orientation = [-5, -10]
            torques = [-0.0001, -0.0002]
            dof_vel = [-0.0001, -0.0005]
            dof_acc = [-1e-6, -1.5e-6]
            base_height = -150
            feet_slip = [-0.0, -0.0]  # Before feet_air_time
            feet_air_time = [1.0, 1.5]
            collision = -1.
            feet_stumble = -1
            action_rate = -0.01
            stand_still = -3  # May affect spot turning
            dof_pos_limits = -0.1
            dof_vel_limits = -1.
            torque_limits = -0.01
            feet_contact_forces = -0.01
            shank_vertical = -2
            # feet_async = -10
            # feet_sync = -10

    class commands(El4090SpiderCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges(El4090SpiderCfg.commands.ranges):
            lin_vel_x = [-1.5, 1.5]  # min max [m/s]
            lin_vel_y = [-1, 1]   # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5]    # min max [rad/s]
            heading = [-1, 1]

    class domain_rand(El4090SpiderCfg.domain_rand):
        # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        randomize_friction = True
        friction_range = [0.3, 1.25]
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        push_robots = True
        push_interval_s = 3
        max_push_vel_xy = 1.

    class noise(El4090SpiderCfg.noise):
        add_noise = True
        noise_level = 1.5  # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 1.5
            lin_vel = 0.8
            ang_vel = 0.8
            gravity = 0.5
            height_measurements = 0.1

    class safety:
        # ------------------------------------------------------------------
        # 开关
        # ------------------------------------------------------------------
        enable_atacom        = True
        clip_nominal_actions = True   # 是否在送入 ATACOM 前裁剪名义动作
        warmup_steps         = 0      # 前 N 步跳过 ATACOM（用于调试）

        # ------------------------------------------------------------------
        # ATACOM 算法超参数
        # ------------------------------------------------------------------
        lambda_retract = 1.0    # 收缩增益 λ：控制向约束流形收缩的速率
        beta           = 2.0    # 松弛变量动力学系数
        dt             = 0.01   # 控制步长（s），建议与仿真 dt 保持一致

        # ------------------------------------------------------------------
        # 关节限位（列表长度须为 18）
        # ------------------------------------------------------------------
        q_max   = [2.95]  * 18    # 关节位置上限（rad）
        q_min   = [-2.95] * 18    # 关节位置下限（rad）
        dq_max  = [14.2]  * 18    # 关节速度上限（rad/s）
        tau_max = [76] * 18    # 关节力矩上限（N·m）

        # ------------------------------------------------------------------
        # 机身限位
        # ------------------------------------------------------------------
        # 三轴倾角上限（rad），对应机体系欧拉角约束
        phi_max = [0.14, 0.14, 0.24] # roll, pitch, yaw 上限（rad）

        z_min = 0.2    # 机身高度下限（m）
        z_max = 0.8   # 机身高度上限（m）

        # ------------------------------------------------------------------
        # 调试
        # ------------------------------------------------------------------
        log_info = False    # 是否每步将 ATACOM info 写入 extras

class El4090SafeCfgPPO( El4090SpiderCfgPPO ):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.3
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'el_4090_safe'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt