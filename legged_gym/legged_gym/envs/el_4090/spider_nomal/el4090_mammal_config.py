from legged_gym.envs.el_4090.spider_nomal.el4090_spider_config import (
    El4090SpiderCfg,
    El4090SpiderCfgPPO,
)


class El4090MammalCfg(El4090SpiderCfg):
    """EL4090 lateral left-right alternating gait inspired by mammal locomotion."""

    class commands(El4090SpiderCfg.commands):
        resampling_time = 4.5

        class ranges(El4090SpiderCfg.commands.ranges):
            lin_vel_x = [-1.2, 1.2]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.6, 1.6]

    class rewards(El4090SpiderCfg.rewards):
        mammal_haa_target = 1.57
        mammal_haa_guidance_ema = 0.01

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
            haa_guidance_mammal = -0.5
            # stand_on_six_legs = -0.15


class El4090MammalCfgPPO(El4090SpiderCfgPPO):
    class runner(El4090SpiderCfgPPO.runner):
        experiment_name = "el4090_mammal"
