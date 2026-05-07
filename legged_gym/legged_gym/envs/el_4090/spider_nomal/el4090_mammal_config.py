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
        class scales(El4090SpiderCfg.rewards.scales):
            gait_2_step = 0.0
            gait_3_step = 0.0
            gait_mammal = -0.8
            stand_on_six_legs = -0.15


class El4090MammalCfgPPO(El4090SpiderCfgPPO):
    class runner(El4090SpiderCfgPPO.runner):
        experiment_name = "el4090_mammal"
