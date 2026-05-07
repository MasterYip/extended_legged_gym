from legged_gym.envs.el_4090.spider_nomal.el4090_spider_config import (
    El4090SpiderCfg,
    El4090SpiderCfgPPO,
)


class El4090JumpCfg(El4090SpiderCfg):
    """EL4090 synchronized hopping / jumping behavior."""

    class commands(El4090SpiderCfg.commands):
        resampling_time = 5.0

        class ranges(El4090SpiderCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.0, 1.0]

    class rewards(El4090SpiderCfg.rewards):
        base_height_target = 0.50
        jump_target_vertical_velocity = 0.9

        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -3.0
            ang_vel_xy = -0.2
            orientation = [-4.0, -2.0]
            torques = -0.0001
            dof_vel = [-0.0002, -0.0004]
            dof_acc = [-5e-8, -1.5e-7]
            base_height = [-1.5, -0.4]
            feet_slip = [-0.0, -0.2]
            feet_air_time = [1.5, 1.0]
            collision = -1.0
            feet_stumble = [-0.0, -0.2]
            action_rate = [-0.005, -0.005]
            stand_still2 = -0.6
            dof_pos_limits = -1.0
            feet_contact_forces = [-0.1, -0.5]
            shank_perp2ground = -0.05
            gait_2_step = 0.0
            gait_3_step = 0.0
            jump_sync = -0.8
            jump_takeoff = -0.8
            stand_on_six_legs = -0.05


class El4090JumpCfgPPO(El4090SpiderCfgPPO):
    class runner(El4090SpiderCfgPPO.runner):
        experiment_name = "el4090_jump"
