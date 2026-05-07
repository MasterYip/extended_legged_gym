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

        class scales(El4090SpiderCfg.rewards.scales):
            gait_2_step = 0.0
            gait_3_step = 0.0
            jump_sync = -0.8
            jump_takeoff = -0.8
            feet_air_time = [1.5, 1.0]
            base_height = [-1.5, -0.4]
            orientation = [-4.0, -2.0]
            stand_on_six_legs = -0.05


class El4090JumpCfgPPO(El4090SpiderCfgPPO):
    class runner(El4090SpiderCfgPPO.runner):
        experiment_name = "el4090_jump"
