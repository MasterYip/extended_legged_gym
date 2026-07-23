from legged_gym.envs.el_4090.spider_nomal.el4090_spider_config import (
    El4090SpiderCfg,
    El4090SpiderCfgPPO,
)


class El4090WaveCfg(El4090SpiderCfg):
    """EL4090 slow wave gait for maximally stable stepping."""

    class commands(El4090SpiderCfg.commands):
        resampling_time = 6.0

        class ranges(El4090SpiderCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-0.8, 0.8]

    class rewards(El4090SpiderCfg.rewards):
        wave_period = 0.72
        wave_clearance = 0.04

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
            base_height = [-2.5, -1.0]
            feet_slip = [-0.1, -0.3]
            feet_air_time = [0.5, 0.1]
            collision = -1.0
            feet_stumble = [-0.0, -0.2]
            action_rate = [-0.005, -0.005]
            stand_still2 = -0.6
            dof_pos_limits = -1.0
            feet_contact_forces = [-0.1, -0.5]
            shank_perp2ground = -0.05
            gait_2_step = 0.0
            gait_3_step = 0.0
            gait_wave = -1.0
            stand_on_six_legs = -0.3


class El4090WaveCfgPPO(El4090SpiderCfgPPO):
    class runner(El4090SpiderCfgPPO.runner):
        experiment_name = "el4090_wave"
