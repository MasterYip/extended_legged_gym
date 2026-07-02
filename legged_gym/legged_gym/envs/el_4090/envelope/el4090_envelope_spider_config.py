# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.el_4090.spider_nomal.el4090_spider_config import El4090SpiderCfg, El4090SpiderCfgPPO


class El4090EnvelopeSpiderCfg(El4090SpiderCfg):
    """Configuration for EL_4090 envelope-aware training environment.

    Extends the base EL_4090 spider config with envelope visualization
    and analysis settings.
    """

    class env(El4090SpiderCfg.env):
        num_observations = 66
        # Debug / visualization
        debug_mode = False
        debug_interval = 100
        debug_env_id = 0
        # Envelope visualization
        enable_envelope_vis = True       # toggle envelope rendering
        envelope_vis_interval = 1         # draw envelope every N steps (1 = every step)

    class envelope:
        """Envelope calculation parameters.

        The hexagonal-prism envelope is defined by:
          - 2D hexagon in the XY plane computed from the six foot positions.
          - Bottom face at min foot height (clamped to min_height).
          - Top face at base_height + height_bias.
        """
        height_bias = 0.00               # [m] offset above base for top face
        min_height = 0.0                  # [m] minimum Z for bottom face
        max_height = None                 # [m] optional cap for top face (None = no cap)
        hexagon_radius_scale = 1.05       # scale factor for hexagon safety margin (>1 = padding)

    class viewer(El4090SpiderCfg.viewer):
        ref_env = 0
        pos = [3.0, -1.0, 2.0]           # camera position
        lookat = [0.5, 0., 0.3]          # camera look-at point


class El4090EnvelopeSpiderCfgPPO(El4090SpiderCfgPPO):
    """PPO-specific config for EL_4090 envelope-aware training."""

    class runner(El4090SpiderCfgPPO.runner):
        run_name = ''
        experiment_name = 'el4090_spider_envelope'
        load_run = -1
        max_iterations = 8000
        multi_stage_rewards = True
