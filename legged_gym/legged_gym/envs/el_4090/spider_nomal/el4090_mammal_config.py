from legged_gym.envs.el_4090.spider_nomal.el4090_spider_config import (
    El4090SpiderCfg,
    El4090SpiderCfgPPO,
)


class El4090MammalCfg(El4090SpiderCfg):
    """EL4090 lateral left-right alternating gait inspired by mammal locomotion."""

    class init_state(El4090SpiderCfg.init_state):
        pos = [0.0, 0.0, 0.45]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # HAA at pi/2 so the network operates in mammal-stance space
            "RF_HAA": -1.3,
            "RM_HAA": 1.3,
            "RB_HAA": 1.3,
            "LF_HAA": -1.3,
            "LM_HAA": 1.3,
            "LB_HAA": 1.3,

            "RF_HFE": 0.6,
            "RM_HFE": 0.6,
            "RB_HFE": 0.6,
            "LF_HFE": 0.6,
            "LM_HFE": 0.6,
            "LB_HFE": 0.6,

            "RF_KFE": 0.0,
            "RM_KFE": 0.0,
            "RB_KFE": 0.0,
            "LF_KFE": 0.0,
            "LM_KFE": 0.0,
            "LB_KFE": 0.0,
        }

    class commands(El4090SpiderCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        small_command_radio = True

        class ranges(El4090SpiderCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.8, 0.8]
            ang_vel_yaw = [-1.5, 1.5]

    class rewards(El4090SpiderCfg.rewards):
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -3.0
            ang_vel_xy = -0.2
            orientation = [-5.0, -3.0]
            torques = -0.0001
            dof_vel = [-0.0002, -0.0004]
            dof_acc = [-5e-8, -1.5e-7]
            base_height = [-2.0, -0.4]
            feet_slip = [-0.0, -0.2]  # Before feet_air_time
            feet_air_time = [0.5, 0.1]
            collision = -1.
            feet_stumble = [-0.0, -0.2]
            action_rate = [-0.005, -0.005]
            stand_still2 = -0.6  # May affect spot turning
            dof_pos_limits = -1.0
            feet_contact_forces = [-0.1, -0.5]

            shank_perp2ground = -0.05
            gait_2_step = [-0.5, -0.0]
            # stand_on_six_legs = -0.15


class El4090MammalCfgPPO(El4090SpiderCfgPPO):
    class runner(El4090SpiderCfgPPO.runner):
        experiment_name = "el4090_mammal"
