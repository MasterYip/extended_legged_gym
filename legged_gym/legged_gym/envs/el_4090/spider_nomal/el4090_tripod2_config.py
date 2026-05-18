from legged_gym.envs.el_4090.spider_nomal.el4090_spider_config import (
    El4090SpiderCfg,
    El4090SpiderCfgPPO,
)


class El4090Tripod2Cfg(El4090SpiderCfg):
    """EL4090 2-group tripod gait on flat terrain."""

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
            feet_slip = [-0.0, -0.2]
            feet_air_time = [0.5, 0.1]
            collision = -1.0
            feet_stumble = [-0.0, -0.2]
            action_rate = [-0.005, -0.005]
            stand_still2 = -0.3
            dof_pos_limits = -1.0
            feet_contact_forces = [-0.1, -0.5]
            shank_perp2ground = -0.05
            gait_2_step = [-0.5, -0.0]
            gait_3_step = 0.0

    class commands(El4090SpiderCfg.commands):
        curriculum = True
        max_curriculum = 3.0
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        small_command_radio = False

        class ranges(El4090SpiderCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.2, 1.2]   # min max [m/s]
            ang_vel_yaw = [-2.0, 2.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

class El4090Tripod2CfgPPO(El4090SpiderCfgPPO):
    class runner(El4090SpiderCfgPPO.runner):
        experiment_name = "el4090_tripod2"
