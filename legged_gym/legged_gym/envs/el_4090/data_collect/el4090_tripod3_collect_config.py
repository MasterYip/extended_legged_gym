from legged_gym.envs.el_4090.spider_nomal.el4090_tripod3_config import (
    El4090Tripod3Cfg,
    El4090Tripod3CfgPPO,
)


class El4090Tripod3CollectCfg(El4090Tripod3Cfg):
    class collect:
        task_vec = [1.0, 1.0, 0.0]


class El4090Tripod3CollectCfgPPO(El4090Tripod3CfgPPO):
    class runner(El4090Tripod3CfgPPO.runner):
        experiment_name = "el4090_tripod3_collect"
