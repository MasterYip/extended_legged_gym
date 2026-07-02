# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from typing import Tuple, List, Optional


class EnvelopeCalculator:
    """Compute a hexagonal-prism envelope for a hexapod robot.

    The envelope is defined as a hexagonal prism:
      - 2D hexagon in the XY plane, computed from the 6 foot positions.
      - Height determined by a bottom reference (ground or min foot Z) and
        a top reference (base height + bias).

    The hexagon vertices are the 6 foot XY positions ordered by angle
    around their centroid, ensuring a convex (or near-convex) simple polygon.

    Usage::

        calc = EnvelopeCalculator(height_bias=0.30, min_height=0.0)
        envelope = calc.compute(foot_positions, base_pos)
        # envelope is a dict with keys: bottom_vertices, top_vertices, edges, ...
    """

    def __init__(self,
                 height_bias: float = 0.30,
                 min_height: float = 0.0,
                 max_height: Optional[float] = None,
                 hexagon_radius_scale: float = 1.05):
        """
        Args:
            height_bias: Offset added to base Z to define the top face height [m].
            min_height: Minimum Z for the bottom face of the prism [m].
            max_height: If set, cap the top face Z at this value [m].
            hexagon_radius_scale: Scale factor applied to radial distances
                when computing the hexagon vertices.  > 1.0 adds a margin.
        """
        self.height_bias = height_bias
        self.min_height = min_height
        self.max_height = max_height
        self.hexagon_radius_scale = hexagon_radius_scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self,
                foot_positions: torch.Tensor,
                base_pos: torch.Tensor) -> dict:
        """Compute the hexagonal-prism envelope for one or more environments.

        Args:
            foot_positions: Tensor [num_envs, 6, 3] -- world-frame XYZ of each foot.
            base_pos: Tensor [num_envs, 3] -- world-frame XYZ of the robot base.

        Returns:
            dict with keys:
                bottom_vertices  -- [num_envs, 6, 3] bottom hexagon corners
                top_vertices     -- [num_envs, 6, 3] top hexagon corners
                bottom_height    -- [num_envs] Z of the bottom face
                top_height       -- [num_envs] Z of the top face
                hex_center       -- [num_envs, 2] (x, y) of the hexagon centre
        """
        num_envs = foot_positions.shape[0]

        # -- 2D hexagon centre ------------------------------------------------
        centre_xy = foot_positions[:, :, :2].mean(dim=1)  # [num_envs, 2]

        # -- ordered hexagon vertices in XY -----------------------------------
        hex_xy = self._compute_hexagon_vertices(foot_positions, centre_xy)

        # -- heights -----------------------------------------------------------
        base_z = base_pos[:, 2]                    # [num_envs]
        foot_z_min = foot_positions[:, :, 2].min(dim=1).values  # [num_envs]

        bottom_z = torch.clamp(foot_z_min, min=self.min_height)
        top_z = base_z + self.height_bias
        if self.max_height is not None:
            top_z = torch.clamp(top_z, max=self.max_height)

        # -- assemble 3D vertices ---------------------------------------------
        bottom_vertices = torch.cat([
            hex_xy,
            bottom_z.unsqueeze(-1).unsqueeze(-1).expand(-1, 6, 1),
        ], dim=-1)  # [num_envs, 6, 3]

        top_vertices = torch.cat([
            hex_xy,
            top_z.unsqueeze(-1).unsqueeze(-1).expand(-1, 6, 1),
        ], dim=-1)  # [num_envs, 6, 3]

        return {
            'bottom_vertices': bottom_vertices,
            'top_vertices': top_vertices,
            'bottom_height': bottom_z,
            'top_height': top_z,
            'hex_center': centre_xy,
        }

    def get_edges(self, vertices: torch.Tensor) -> List[Tuple[int, int]]:
        """Return index-pairs defining the edges of a closed N-gon.

        Args:
            vertices: [N, 3] vertices of the polygon.

        Returns:
            List of (i, j) index tuples.
        """
        n = vertices.shape[0]
        edges = [(i, (i + 1) % n) for i in range(n)]
        return edges

    def get_prism_edges(self) -> Tuple[List[Tuple[int, int]],
                                       List[Tuple[int, int]],
                                       List[Tuple[int, int]]]:
        """Return edge index-pairs for the bottom, top, and vertical edges.

        Returns:
            (bottom_edges, top_edges, vertical_edges) where each is a list of
            (i, j) tuples (0-based indices into the 6-element vertex arrays).
        """
        n = 6
        bottom_edges = [(i, (i + 1) % n) for i in range(n)]
        top_edges = [(i, (i + 1) % n) for i in range(n)]
        vertical_edges = [(i, i) for i in range(n)]  # connect bottom[i] to top[i]
        return bottom_edges, top_edges, vertical_edges

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_hexagon_vertices(self,
                                  foot_positions: torch.Tensor,
                                  centre_xy: torch.Tensor) -> torch.Tensor:
        """Order the 6 foot XY positions by angle around *centre_xy*.

        Args:
            foot_positions: [num_envs, 6, 3]
            centre_xy: [num_envs, 2]

        Returns:
            hex_xy: [num_envs, 6, 2] ordered hexagon vertices.
        """
        num_envs = foot_positions.shape[0]
        foot_xy = foot_positions[:, :, :2]  # [num_envs, 6, 2]

        # vectors from centre to each foot
        vecs = foot_xy - centre_xy.unsqueeze(1)  # [num_envs, 6, 2]

        # angle in [-pi, pi]
        angles = torch.atan2(vecs[:, :, 1], vecs[:, :, 0])  # [num_envs, 6]

        # sort each env's feet by angle
        _, idx = torch.sort(angles, dim=1)  # [num_envs, 6]

        # gather the ordered XY positions
        ordered_xy = torch.gather(
            foot_xy, 1,
            idx.unsqueeze(-1).expand(-1, -1, 2),
        )  # [num_envs, 6, 2]

        # Optionally scale outward from centre to add a safety margin
        if self.hexagon_radius_scale != 1.0:
            radial = ordered_xy - centre_xy.unsqueeze(1)
            ordered_xy = centre_xy.unsqueeze(1) + radial * self.hexagon_radius_scale

        return ordered_xy


def hexagon_vertices_from_center(center_xy: np.ndarray,
                                  radius: float,
                                  heading: float = 0.0) -> np.ndarray:
    """Generate 6 vertices of a regular hexagon centred at *center_xy*.

    This is a convenience helper that does NOT depend on foot positions.
    Useful for drawing a fixed reference hexagon.

    Args:
        center_xy: (2,) array [x, y].
        radius: circumradius of the hexagon [m].
        heading: rotation offset [rad] (default: 0, first vertex at angle 0).

    Returns:
        vertices: (6, 2) array of XY positions.
    """
    angles = heading + np.linspace(0, 2 * np.pi, 7)[:6]  # 6 equally-spaced angles
    x = center_xy[0] + radius * np.cos(angles)
    y = center_xy[1] + radius * np.sin(angles)
    return np.stack([x, y], axis=-1)
