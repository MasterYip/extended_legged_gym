from legged_gym.envs.el_4090.spider_nomal.el4090_jump_config import (
    El4090JumpCfg,
    El4090JumpCfgPPO,
)


class El4090JumpCollectCfg(El4090JumpCfg):
    class collect:
        task_vec = [1.0, 3.0, 0.0]


class El4090JumpCollectCfgPPO(El4090JumpCfgPPO):
    class runner(El4090JumpCfgPPO.runner):
        experiment_name = "el4090_jump_collect"
