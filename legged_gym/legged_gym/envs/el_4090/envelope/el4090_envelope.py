# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaacgym.torch_utils import *

import torch
from typing import Optional, List

from legged_gym.envs.el_4090.spider_nomal.el_4090 import EL_4090
from legged_gym.envs.el_4090.envelope.el4090_envelope_spider_config import El4090EnvelopeSpiderCfg
from legged_gym.utils.envelope import EnvelopeCalculator
from legged_gym import LEGGED_GYM_ROOT_DIR


class EL_4090_Envelope(EL_4090):
    """EL_4090 environment with hexagonal-prism envelope visualization and analysis.

    The envelope is a hexagonal prism defined by:
      - A 2D hexagon in the XY plane computed from **all** body positions
        matching ``envelope_body_filters`` (default: ``['FOOT']``).  Add
        ``'SHANK'``, ``'THIGH'``, ``'HIP'`` to enclose protruding joints.
      - Bottom face: at min body Z (clamped to *min_height*).
      - Top face: at ``base_height + height_bias``.

    Visualization is drawn via ``GymVisualizer`` during training when
    ``enable_envelope_vis`` is True in the config.

    **Body-name to joint mapping** (EL_4090 rigid bodies)::

        Body substring   Roughly corresponds to
        ───────────────  ──────────────────────
        FOOT             Foot endpoint
        SHANK            KFE joint (knee)
        THIGH            HFE joint (hip pitch)
        HIP              HAA joint (hip roll)

    Usage::

        from legged_gym.envs.el_4090.envelope.el4090_envelope import EL_4090_Envelope
        from legged_gym.envs.el_4090.envelope.el4090_envelope_spider_config import El4090EnvelopeSpiderCfg

        env = EL_4090_Envelope(cfg=El4090EnvelopeSpiderCfg(), ...)
    """

    cfg: El4090EnvelopeSpiderCfg

    def __init__(self, cfg: El4090EnvelopeSpiderCfg, sim_params, physics_engine,
                 sim_device, headless, task_name="el4090_spider_envelope"):
        # Parse envelope config BEFORE super().__init__ because _init_buffers()
        # (called during super) needs self._envelope_body_filters.
        envelope_cfg = getattr(cfg, 'envelope', None)
        if envelope_cfg is not None:
            height_bias = getattr(envelope_cfg, 'height_bias', 0.30)
            min_height = getattr(envelope_cfg, 'min_height', 0.0)
            max_height = getattr(envelope_cfg, 'max_height', None)
            radius_scale = getattr(envelope_cfg, 'hexagon_radius_scale', 1.05)
            body_filters: List[str] = list(getattr(envelope_cfg, 'envelope_body_filters', ['FOOT']))
        else:
            height_bias = 0.30
            min_height = 0.0
            max_height = None
            radius_scale = 1.05
            body_filters = ['FOOT']

        self._envelope_body_filters = body_filters
        self._envelope_body_indices: Optional[torch.Tensor] = None
        self._envelope_step_counter = 0
        self._last_envelope: Optional[dict] = None

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, task_name=task_name)

        self.envelope_calculator = EnvelopeCalculator(
            height_bias=height_bias,
            min_height=min_height,
            max_height=max_height,
            hexagon_radius_scale=radius_scale,
        )

        self._enable_envelope_vis = getattr(cfg.env, 'enable_envelope_vis', True)
        self._envelope_vis_interval = getattr(cfg.env, 'envelope_vis_interval', 1)

        # Enable debug_viz so that _draw_debug_vis is called in post_physics_step
        if self._enable_envelope_vis and not self.headless:
            self.debug_viz = True

    # ------------------------------------------------------------------
    # Buffer init
    # ------------------------------------------------------------------

    def _init_buffers(self):
        """Extend parent to build the envelope body-index tensor."""
        super()._init_buffers()

        # Retrieve body names from the first actor (envs are already created)
        body_names = self.gym.get_actor_rigid_body_names(
            self.envs[0], self.actor_handles[0],
        )

        # Collect indices of rigid bodies whose name contains any filter
        idx_list = []
        for filter_str in self._envelope_body_filters:
            for i, name in enumerate(body_names):
                if filter_str in name:
                    idx_list.append(i)
        # Deduplicate while preserving order
        seen = set()
        idx_list = [i for i in idx_list if not (i in seen or seen.add(i))]

        if len(idx_list) == 0:
            # Fallback to feet only
            idx_list = self.feet_indices.tolist()

        self._envelope_body_indices = torch.tensor(
            idx_list, dtype=torch.long, device=self.device, requires_grad=False,
        )

        print(f"[EL_4090_Envelope] envelope body filters: {self._envelope_body_filters}")
        print(f"[EL_4090_Envelope] matched {len(idx_list)} bodies: "
              f"{[body_names[i] for i in idx_list]}")

    # ------------------------------------------------------------------
    # Public envelope API
    # ------------------------------------------------------------------

    def compute_envelope(self) -> dict:
        """Compute the hexagonal-prism envelope for all environments.

        Uses the positions of all bodies matching ``envelope_body_filters``
        (not just feet) and the base position.

        Returns:
            dict with keys:
                bottom_vertices  -- [num_envs, 6, 3]
                top_vertices     -- [num_envs, 6, 3]
                bottom_height    -- [num_envs]
                top_height       -- [num_envs]
                hex_center       -- [num_envs, 2]
        """
        # Gather world-frame positions of all envelope-relevant bodies
        body_pos = self._get_envelope_body_positions()  # [num_envs, N, 3]
        base_pos = self.base_pos                         # [num_envs, 3]

        envelope = self.envelope_calculator.compute(body_pos, base_pos)
        self._last_envelope = envelope
        return envelope

    def get_envelope_volume(self) -> torch.Tensor:
        """Return the volume of the hexagonal prism for each environment [num_envs].

        Volume of a regular hexagonal prism:
            V = (3 * sqrt(3) / 2) * r^2 * h
        where *r* is the mean distance from centre to hexagon vertices and
        *h* is the prism height.
        """
        envelope = self._last_envelope
        if envelope is None:
            envelope = self.compute_envelope()

        centre = envelope['hex_center']       # [num_envs, 2]
        bottom = envelope['bottom_vertices']   # [num_envs, 6, 3]
        top_z = envelope['top_height']         # [num_envs]
        bottom_z = envelope['bottom_height']   # [num_envs]

        # mean radius from centre
        radial = torch.norm(bottom[:, :, :2] - centre.unsqueeze(1), dim=-1)  # [num_envs, 6]
        mean_radius = radial.mean(dim=1)  # [num_envs]

        height = top_z - bottom_z  # [num_envs]

        # area of regular hexagon: (3√3 / 2) * r²
        area = (3.0 * math.sqrt(3.0) / 2.0) * (mean_radius ** 2)
        volume = area * torch.clamp(height, min=0.0)
        return volume

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_envelope_body_positions(self) -> torch.Tensor:
        """Extract world-frame XYZ of all bodies matching the name filters.

        Returns:
            body_pos: [num_envs, N, 3] where N = len(self._envelope_body_indices).
        """
        # rigid_body_state: [num_envs * num_bodies, 13]
        rb_state = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13,
        )
        indices = self._envelope_body_indices  # [N]
        # Gather by index: [num_envs, N, 13] → slice positions [:, :, 0:3]
        body_pos = rb_state[:, indices, 0:3]   # [num_envs, N, 3]
        return body_pos

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _draw_debug_vis(self):
        """Override to add envelope visualization after parent visuals."""
        # Draw parent debug visuals first (base vel arrows)
        super()._draw_debug_vis()

        if not self._enable_envelope_vis:
            return

        self._envelope_step_counter += 1
        if self._envelope_step_counter % self._envelope_vis_interval != 0:
            return

        # Refresh rigid body state so positions are current
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Compute envelope from current state
        envelope = self.compute_envelope()

        bottom_vertices = envelope['bottom_vertices']  # [num_envs, 6, 3]
        top_vertices = envelope['top_vertices']        # [num_envs, 6, 3]

        bottom_edges, top_edges, vertical_edges = \
            self.envelope_calculator.get_prism_edges()

        # Draw only the reference environment for performance
        ref_env = getattr(self.cfg.viewer, 'ref_env', 0)
        env_indices = [ref_env] if ref_env < self.num_envs else [0]

        for env_idx in env_indices:
            bottom_np = bottom_vertices[env_idx].cpu().numpy()  # [6, 3]
            top_np = top_vertices[env_idx].cpu().numpy()        # [6, 3]

            # -- Bottom hexagon (cyan) -----------------------------------------
            for i, j in bottom_edges:
                self.vis.draw_boldline(
                    env_idx,
                    [bottom_np[i], bottom_np[j]],
                    rad=0.01,
                    resolution=8,
                    color=(0.0, 1.0, 1.0),
                )

            # -- Top hexagon (magenta) ------------------------------------------
            for i, j in top_edges:
                self.vis.draw_boldline(
                    env_idx,
                    [top_np[i], top_np[j]],
                    rad=0.01,
                    resolution=8,
                    color=(1.0, 0.0, 1.0),
                )

            # -- Vertical edges (yellow) ----------------------------------------
            for i, _ in vertical_edges:
                self.vis.draw_boldline(
                    env_idx,
                    [bottom_np[i], top_np[i]],
                    rad=0.01,
                    resolution=8,
                    color=(1.0, 1.0, 0.0),
                )

            # -- Centre marker (small white sphere) -----------------------------
            centre_xy = envelope['hex_center'][env_idx].cpu().numpy()
            centre_z = (envelope['bottom_height'][env_idx].item() +
                         envelope['top_height'][env_idx].item()) / 2.0
            self.vis.draw_point(
                env_idx,
                [centre_xy[0], centre_xy[1], centre_z],
                color=(1.0, 1.0, 1.0),
                size=0.03,
            )

            # -- Envelope body markers (small spheres per filtered body) --------
            body_pos_np = self._get_envelope_body_positions()[env_idx].cpu().numpy()
            for bp in body_pos_np:
                self.vis.draw_point(
                    env_idx,
                    bp,
                    color=(1.0, 0.65, 0.0),   # orange
                    size=0.02,
                )

    def _draw_envelope_info_overlay(self):
        """Log textual envelope information for debugging.

        Printed to stdout at *envelope_vis_interval* frequency when
        *debug_mode* is enabled.
        """
        envelope = self._last_envelope
        if envelope is None:
            envelope = self.compute_envelope()

        volume = self.get_envelope_volume()
        env_id = self.cfg.env.debug_env_id

        if env_id >= self.num_envs:
            return

        print("\n" + "=" * 60)
        print(f"[Envelope Info] Step {self.common_step_counter} | Env {env_id}")
        print("-" * 60)
        print(f"  Bottom Z:      {envelope['bottom_height'][env_id].item():.3f} m")
        print(f"  Top Z:         {envelope['top_height'][env_id].item():.3f} m")
        print(f"  Height:        {envelope['top_height'][env_id].item() - envelope['bottom_height'][env_id].item():.3f} m")
        print(f"  Centre XY:     [{envelope['hex_center'][env_id, 0].item():.3f}, "
              f"{envelope['hex_center'][env_id, 1].item():.3f}]")
        print(f"  Volume:        {volume[env_id].item():.4f} m^3")
        print(f"  Num bodies:    {len(self._envelope_body_indices)}")
        print("=" * 60 + "\n")
