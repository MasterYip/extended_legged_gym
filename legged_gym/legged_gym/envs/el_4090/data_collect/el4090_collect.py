from legged_gym.envs.el_4090.spider_nomal.el_4090 import EL_4090
from legged_gym.envs.elspider_air.elspider_data_collect import ElSpiderDataCollect


class EL4090DataCollect(EL_4090, ElSpiderDataCollect):
    """EL_4090 variant that exposes ``get_diffusion_observation()`` for offline
    data collection.

    MRO: EL4090DataCollect → EL_4090 → ElSpiderDataCollect → ElSpider → LeggedRobot

    All state tensors are already set up by EL_4090 (and its parents) through
    cooperative super() calls, so no additional overrides are needed here.
    """
