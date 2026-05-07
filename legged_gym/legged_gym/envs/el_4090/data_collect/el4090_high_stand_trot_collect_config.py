from legged_gym.envs.el_4090.spider_nomal.el4090_high_stand_trot_config import (
    El4090HighStandTrotCfg,
    El4090HighStandTrotCfgPPO,
)


class El4090HighStandTrotCollectCfg(El4090HighStandTrotCfg):
    class collect:
        task_vec = [1.0, 4.0, 0.0]


class El4090HighStandTrotCollectCfgPPO(El4090HighStandTrotCfgPPO):
    class runner(El4090HighStandTrotCfgPPO.runner):
        experiment_name = "el4090_high_stand_trot_collect"
