"""Envelope state kept outside locomotion commands and policy observations."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from legged_gym.utils.envelop.network.haa_swing_range import (
    EnvelopeConditionSpec,
    apply_env_morphology_priors,
)


class EnvelopeConditionState:
    """Own, sample and externally update batched envelope conditions.

    The locomotion command tensor remains ``[vx, vy, yaw_rate]``.  This class
    independently stores the eight values required by morphology and the HAA
    range network, making it possible to replace random sampling with a future
    perception/planning module without changing the policy observation layout.
    """

    def __init__(self, cfg, num_envs: int, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.condition_names = tuple(cfg.condition_names)
        self.condition_dim = len(self.condition_names)
        self.low = torch.tensor(
            [getattr(cfg.ranges, name)[0] for name in self.condition_names],
            dtype=torch.float,
            device=self.device,
        )
        self.high = torch.tensor(
            [getattr(cfg.ranges, name)[1] for name in self.condition_names],
            dtype=torch.float,
            device=self.device,
        )
        self.spec = EnvelopeConditionSpec(
            condition_names=self.condition_names,
            low=tuple(float(value) for value in self.low.cpu().tolist()),
            high=tuple(float(value) for value in self.high.cpu().tolist()),
            morphology_prior_mode=str(cfg.morphology_prior_mode),
            morphology_prior_weights=cfg.morphology_prior_weights,
            morphology_middle_front_follow_weight=float(
                cfg.morphology_middle_front_follow_weight
            ),
        )
        midpoint = 0.5 * (self.low + self.high)
        initial = apply_env_morphology_priors(midpoint.unsqueeze(0), self.spec)[0]
        self.condition = initial.repeat(self.num_envs, 1)

    def get(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.condition if env_ids is None else self.condition[env_ids]

    def sample(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample envelope bounds, then derive priors exactly as training does."""
        if env_ids.numel() == 0:
            return self.condition[env_ids]
        random_values = torch.rand(
            (env_ids.numel(), self.condition_dim),
            dtype=self.condition.dtype,
            device=self.device,
        )
        sampled = self.low + random_values * (self.high - self.low)
        sampled = apply_env_morphology_priors(sampled, self.spec)
        self.condition[env_ids] = sampled
        return sampled

    def set(
        self,
        values: torch.Tensor | Sequence[Sequence[float]],
        env_ids: Optional[torch.Tensor] = None,
        *,
        derive_priors: bool = True,
    ) -> torch.Tensor:
        """Set externally computed envelopes for all or selected environments."""
        target_ids = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None
            else env_ids.to(device=self.device, dtype=torch.long)
        )
        tensor = torch.as_tensor(values, dtype=torch.float, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape != (target_ids.numel(), self.condition_dim):
            raise ValueError(
                f"Expected envelope shape {(target_ids.numel(), self.condition_dim)}, "
                f"got {tuple(tensor.shape)}"
            )
        tensor = torch.minimum(torch.maximum(tensor, self.low), self.high)
        if derive_priors:
            tensor = apply_env_morphology_priors(tensor, self.spec)
        self.condition[target_ids] = tensor
        return tensor
