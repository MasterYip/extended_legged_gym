from legged_gym.envs.elspider_air.flat.elspider_air_flat_config import (
    ElSpiderAirFlatCfg,
    ElSpiderAirFlatCfgPPO,
)


class ElSpiderFlatCollectCfg(ElSpiderAirFlatCfg):
    """Config for elspider flat-walk data collection.

    Inherits verbatim from ``ElSpiderAirFlatCfg`` to guarantee that the policy
    observation space is identical to the one used during training.

    task_vec semantics (length-3, consistent across all collect configs):
        [robot_id, gait_id, terrain_id]
        ElSpider flat walk: robot=0 (elspider_air), gait=0 (2-step), terrain=0 (flat)
    """

    class collect:
        task_vec = [0.0, 0.0, 0.0]


class ElSpiderFlatCollectCfgPPO(ElSpiderAirFlatCfgPPO):
    class runner(ElSpiderAirFlatCfgPPO.runner):
        experiment_name = "elspider_flat_collect"
