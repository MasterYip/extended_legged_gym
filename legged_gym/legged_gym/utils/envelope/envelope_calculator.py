# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
import torch
from typing import Tuple, List, Optional


class EnvelopeCalculator:
    """Compute a hexagonal-prism envelope for a legged robot.

    The envelope is defined as a hexagonal prism:
      - 2D hexagon in the XY plane computed from an arbitrary set of body
        positions (feet, knees, shanks, etc. via name filters).
      - Height determined by a bottom reference (ground or min body Z) and
        a top reference (base height + bias).

    The hexagon vertices are the extreme points in 6 equal angular sectors
    (60° each) around the centroid of all body XY positions.  This ensures
    the hexagon encompasses every included body part even when N ≠ 6.

    Usage::

        calc = EnvelopeCalculator(height_bias=0.30, min_height=0.0)
        envelope = calc.compute(body_positions, base_pos)
    """

    NUM_HEX_VERTICES = 6

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
                body_positions: torch.Tensor,
                base_pos: torch.Tensor) -> dict:
        """Compute the hexagonal-prism envelope for one or more environments.

        Args:
            body_positions: Tensor [num_envs, N, 3] -- world-frame XYZ of
                all body parts to enclose (feet, knees, shanks, etc.).
                N can be any value ≥ 1; the hexagon always has 6 vertices.
            base_pos: Tensor [num_envs, 3] -- world-frame XYZ of the robot base.

        Returns:
            dict with keys:
                bottom_vertices  -- [num_envs, 6, 3] bottom hexagon corners
                top_vertices     -- [num_envs, 6, 3] top hexagon corners
                bottom_height    -- [num_envs] Z of the bottom face
                top_height       -- [num_envs] Z of the top face
                hex_center       -- [num_envs, 2] (x, y) of the hexagon centre
        """
        num_envs = body_positions.shape[0]

        # -- 2D hexagon centre ------------------------------------------------
        centre_xy = body_positions[:, :, :2].mean(dim=1)  # [num_envs, 2]

        # -- hexagon vertices in XY (extreme points per angular sector) --------
        hex_xy = self._compute_hexagon_vertices(body_positions, centre_xy)

        # -- heights -----------------------------------------------------------
        base_z = base_pos[:, 2]                          # [num_envs]
        body_z_min = body_positions[:, :, 2].min(dim=1).values  # [num_envs]

        bottom_z = torch.clamp(body_z_min, min=self.min_height)
        top_z = base_z + self.height_bias
        if self.max_height is not None:
            top_z = torch.clamp(top_z, max=self.max_height)

        # -- assemble 3D vertices ---------------------------------------------
        bottom_vertices = torch.cat([
            hex_xy,
            bottom_z.unsqueeze(-1).unsqueeze(-1).expand(-1, self.NUM_HEX_VERTICES, 1),
        ], dim=-1)  # [num_envs, 6, 3]

        top_vertices = torch.cat([
            hex_xy,
            top_z.unsqueeze(-1).unsqueeze(-1).expand(-1, self.NUM_HEX_VERTICES, 1),
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
        return [(i, (i + 1) % n) for i in range(n)]

    def get_prism_edges(self) -> Tuple[List[Tuple[int, int]],
                                       List[Tuple[int, int]],
                                       List[Tuple[int, int]]]:
        """Return edge index-pairs for the bottom, top, and vertical edges.

        Returns:
            (bottom_edges, top_edges, vertical_edges) where each is a list of
            (i, j) tuples (0-based indices into the 6-element vertex arrays).
        """
        n = self.NUM_HEX_VERTICES
        bottom_edges = [(i, (i + 1) % n) for i in range(n)]
        top_edges = [(i, (i + 1) % n) for i in range(n)]
        vertical_edges = [(i, i) for i in range(n)]
        return bottom_edges, top_edges, vertical_edges

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_hexagon_vertices(self,
                                  body_positions: torch.Tensor,
                                  centre_xy: torch.Tensor) -> torch.Tensor:
        """Compute 6 hexagon vertices via angular-sector binning.

        The circle is divided into *NUM_HEX_VERTICES* equal sectors (60° each).
        Within each sector the body point with the **largest** distance from
        *centre_xy* is chosen as the hexagon vertex.  If a sector is empty the
        vertex from the nearest non-empty sector is reused.

        Args:
            body_positions: [num_envs, N, 3]  (N may vary across envs but
                tensor shape is fixed — mask empty slots with the centroid).
            centre_xy: [num_envs, 2]

        Returns:
            hex_xy: [num_envs, 6, 2] ordered hexagon vertices (CCW from -π).
        """
        num_envs = body_positions.shape[0]
        body_xy = body_positions[:, :, :2]  # [num_envs, N, 2]

        # vectors & angles from centre
        vecs = body_xy - centre_xy.unsqueeze(1)      # [num_envs, N, 2]
        dists = torch.norm(vecs, dim=-1)              # [num_envs, N]
        angles = torch.atan2(vecs[:, :, 1], vecs[:, :, 0])  # [num_envs, N]  in [-π, π]

        n_vert = self.NUM_HEX_VERTICES
        sector_width = 2.0 * math.pi / n_vert  # 60°

        # Sector boundaries: n_vert+1 edges from -π to π
        sector_edges = torch.linspace(-math.pi, math.pi, n_vert + 1,
                                       device=body_positions.device)

        hex_xy_list = []
        for s in range(n_vert):
            lo = sector_edges[s]
            hi = sector_edges[s + 1]

            # Mask points in this sector
            if s == n_vert - 1:
                # last sector includes the upper boundary (π)
                in_sector = (angles >= lo) & (angles <= hi)
            else:
                in_sector = (angles >= lo) & (angles < hi)

            # For each env, find the farthest point in this sector
            masked_dists = torch.where(in_sector, dists,
                                        torch.full_like(dists, -1.0))
            max_dist, max_idx = torch.max(masked_dists, dim=1)  # [num_envs]

            # If sector is empty (max_dist < 0), fall back to the centroid
            empty = max_dist < 0.0  # [num_envs] bool
            max_idx = torch.where(empty, torch.zeros_like(max_idx), max_idx)

            # Gather the XY of the farthest point
            vertex_xy = torch.gather(
                body_xy, 1,
                max_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2),
            ).squeeze(1)  # [num_envs, 2]

            # For empty sectors replace with centroid (so scaling still works)
            vertex_xy = torch.where(
                empty.unsqueeze(-1).expand(-1, 2),
                centre_xy,
                vertex_xy,
            )

            hex_xy_list.append(vertex_xy)

        hex_xy = torch.stack(hex_xy_list, dim=1)  # [num_envs, 6, 2]

        # Fill empty-sector vertices by copying from nearest non-empty neighbour
        hex_xy = self._fill_empty_sectors(hex_xy, centre_xy)

        # Optionally scale outward from centre to add a safety margin
        if self.hexagon_radius_scale != 1.0:
            radial = hex_xy - centre_xy.unsqueeze(1)
            hex_xy = centre_xy.unsqueeze(1) + radial * self.hexagon_radius_scale

        return hex_xy

    def _fill_empty_sectors(self,
                            hex_xy: torch.Tensor,
                            centre_xy: torch.Tensor) -> torch.Tensor:
        """Replace any vertex still at the centroid with the nearest non-centroid
        neighbour (wrapping around the hexagon).

        Args:
            hex_xy: [num_envs, 6, 2]
            centre_xy: [num_envs, 2]

        Returns:
            hex_xy with empty slots filled.
        """
        n_vert = hex_xy.shape[1]
        # Identify empty vertices (still at centroid)
        at_centre = torch.all(
            torch.abs(hex_xy - centre_xy.unsqueeze(1)) < 1e-6, dim=-1
        )  # [num_envs, 6]

        # If none empty, return early
        if not at_centre.any():
            return hex_xy

        # For each empty vertex walk outward in both directions to find a
        # non-empty neighbour, then average them.
        result = hex_xy.clone()
        for offset in range(1, n_vert):
            left = (torch.arange(n_vert, device=hex_xy.device) - offset) % n_vert
            right = (torch.arange(n_vert, device=hex_xy.device) + offset) % n_vert

            # Gather left and right neighbours
            left_xy = torch.gather(result, 1,
                                    left.unsqueeze(0).unsqueeze(-1).expand(hex_xy.shape[0], -1, 2))
            right_xy = torch.gather(result, 1,
                                     right.unsqueeze(0).unsqueeze(-1).expand(hex_xy.shape[0], -1, 2))

            # Take whichever is non-empty, or average if both non-empty
            left_empty = torch.all(
                torch.abs(left_xy - centre_xy.unsqueeze(1)) < 1e-6, dim=-1)
            right_empty = torch.all(
                torch.abs(right_xy - centre_xy.unsqueeze(1)) < 1e-6, dim=-1)

            both_valid = ~left_empty & ~right_empty & at_centre
            left_only = ~left_empty & right_empty & at_centre
            right_only = left_empty & ~right_empty & at_centre

            fill_mask = both_valid | left_only | right_only

            fill_xy = torch.zeros_like(result)
            fill_xy = torch.where(
                both_valid.unsqueeze(-1).expand_as(fill_xy),
                (left_xy + right_xy) / 2.0,
                fill_xy,
            )
            fill_xy = torch.where(
                left_only.unsqueeze(-1).expand_as(fill_xy),
                left_xy,
                fill_xy,
            )
            fill_xy = torch.where(
                right_only.unsqueeze(-1).expand_as(fill_xy),
                right_xy,
                fill_xy,
            )

            result = torch.where(
                fill_mask.unsqueeze(-1).expand_as(result),
                fill_xy,
                result,
            )
            at_centre = at_centre & ~fill_mask

            if not at_centre.any():
                break

        return result


def hexagon_vertices_from_center(center_xy: np.ndarray,
                                  radius: float,
                                  heading: float = 0.0) -> np.ndarray:
    """Generate 6 vertices of a regular hexagon centred at *center_xy*.

    This is a convenience helper that does NOT depend on body positions.
    Useful for drawing a fixed reference hexagon.

    Args:
        center_xy: (2,) array [x, y].
        radius: circumradius of the hexagon [m].
        heading: rotation offset [rad] (default: 0, first vertex at angle 0).

    Returns:
        vertices: (6, 2) array of XY positions.
    """
    angles = heading + np.linspace(0, 2 * np.pi, 7)[:6]
    x = center_xy[0] + radius * np.cos(angles)
    y = center_xy[1] + radius * np.sin(angles)
    return np.stack([x, y], axis=-1)
