from legged_gym.envs.el_4090.spider_nomal.el4090_wave_config import (
    El4090WaveCfg,
    El4090WaveCfgPPO,
)


class El4090WaveCollectCfg(El4090WaveCfg):
    class collect:
        task_vec = [1.0, 2.0, 0.0]


class El4090WaveCollectCfgPPO(El4090WaveCfgPPO):
    class runner(El4090WaveCfgPPO.runner):
        experiment_name = "el4090_wave_collect"
