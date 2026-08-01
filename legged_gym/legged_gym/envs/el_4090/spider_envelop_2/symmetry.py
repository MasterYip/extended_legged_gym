"""The observation/action layout is unchanged, so reuse the verified mirror map."""

from legged_gym.envs.el_4090.spider_envelop.symmetry import (  # noqa: F401
    get_elair_lidar_xsym_obs_act,
)

__all__ = ["get_elair_lidar_xsym_obs_act"]
