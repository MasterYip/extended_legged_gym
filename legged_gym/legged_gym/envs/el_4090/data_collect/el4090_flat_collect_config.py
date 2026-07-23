from .el4090_tripod2_collect_config import (
    El4090Tripod2CollectCfg,
    El4090Tripod2CollectCfgPPO,
)


class El4090FlatCollectCfg(El4090Tripod2CollectCfg):
    """Config for EL_4090 flat-walk data collection.

    Inherits verbatim from ``El4090SpiderCfg`` to guarantee that the policy
    observation space is identical to the one used during training.

    task_vec semantics (length-3, consistent across all collect configs):
        [robot_id, gait_id, terrain_id]
        EL4090 flat walk: robot=1 (el4090), gait=0 (tripod-2), terrain=0 (flat)
    """

    class collect:
        task_vec = [1.0, 0.0, 0.0]


class El4090FlatCollectCfgPPO(El4090Tripod2CollectCfgPPO):
    class runner(El4090Tripod2CollectCfgPPO.runner):
        experiment_name = "el4090_flat_collect"
