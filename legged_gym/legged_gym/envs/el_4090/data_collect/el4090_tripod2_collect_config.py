from legged_gym.envs.el_4090.spider_nomal.el4090_tripod2_config import (
    El4090Tripod2Cfg,
    El4090Tripod2CfgPPO,
)


class El4090Tripod2CollectCfg(El4090Tripod2Cfg):
    class collect:
        task_vec = [1.0, 0.0, 0.0]


class El4090Tripod2CollectCfgPPO(El4090Tripod2CfgPPO):
    class runner(El4090Tripod2CfgPPO.runner):
        experiment_name = "el4090_tripod2_collect"
