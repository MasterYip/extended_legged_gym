from legged_gym.envs.el_4090.spider_nomal.el4090_spider_config import (
    El4090SpiderCfg,
    El4090SpiderCfgPPO,
)


class El4090HighStandTrotCfg(El4090SpiderCfg):
    """EL4090 high-standing tripod trot with a taller nominal posture."""

    class init_state(El4090SpiderCfg.init_state):
        pos = [0.0, 0.0, 0.52]

    class rewards(El4090SpiderCfg.rewards):
        base_height_target = 0.56

        class scales(El4090SpiderCfg.rewards.scales):
            gait_2_step = [-0.35, -0.0]
            gait_3_step = 0.0
            base_height = [-3.0, -1.5]
            orientation = [-6.0, -4.0]
            stand_on_six_legs = -0.2


class El4090HighStandTrotCfgPPO(El4090SpiderCfgPPO):
    class runner(El4090SpiderCfgPPO.runner):
        experiment_name = "el4090_high_stand_trot"
