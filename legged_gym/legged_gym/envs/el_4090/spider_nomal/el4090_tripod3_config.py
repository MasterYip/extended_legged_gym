from legged_gym.envs.el_4090.spider_nomal.el4090_spider_config import (
    El4090SpiderCfg,
    El4090SpiderCfgPPO,
)


class El4090Tripod3Cfg(El4090SpiderCfg):
    """EL4090 3-group gait with front/middle/rear synchronization pairs."""

    class rewards(El4090SpiderCfg.rewards):
        class scales(El4090SpiderCfg.rewards.scales):
            gait_2_step = 0.0
            gait_3_step = [-0.5, -0.0]


class El4090Tripod3CfgPPO(El4090SpiderCfgPPO):
    class runner(El4090SpiderCfgPPO.runner):
        experiment_name = "el4090_tripod3"
