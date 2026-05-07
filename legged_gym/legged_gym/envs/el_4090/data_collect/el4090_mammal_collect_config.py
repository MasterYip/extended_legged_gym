from legged_gym.envs.el_4090.spider_nomal.el4090_mammal_config import (
    El4090MammalCfg,
    El4090MammalCfgPPO,
)


class El4090MammalCollectCfg(El4090MammalCfg):
    class collect:
        task_vec = [1.0, 5.0, 0.0]


class El4090MammalCollectCfgPPO(El4090MammalCfgPPO):
    class runner(El4090MammalCfgPPO.runner):
        experiment_name = "el4090_mammal_collect"
