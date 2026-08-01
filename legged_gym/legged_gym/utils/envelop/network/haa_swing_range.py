"""Generate per-leg HAA swing ranges from envelope and morphology priors.

The analytical model treats the lateral envelope as a hard geometric limit:

    hip_offset + leg_reach * sin(|q_haa - q_prior|) <= envelope_width

and intersects that limit with a morphology-dependent swing allowance and the
physical HAA joint limits.  The Monte-Carlo implementation estimates the same
feasible set by sampling.  ``HaaRangeNetwork`` can be trained to distil either
estimator for cheap batched inference in the simulator.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

import torch
from torch import Tensor, nn


DEFAULT_CONDITION_NAMES: Tuple[str, ...] = (
    "front_width",
    "middle_width",
    "back_width",
    "forward_limit",
    "backward_limit",
    "morphology_front_prior",
    "morphology_middle_prior",
    "morphology_back_prior",
)
DEFAULT_LEG_NAMES: Tuple[str, ...] = ("RF", "RM", "RB", "LF", "LM", "LB")
DEFAULT_ENV_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "envs"
    / "el_4090"
    / "spider_envelop"
    / "el4090_spider_config.py"
)


@dataclass(frozen=True)
class EnvelopeConditionSpec:
    """Condition distribution read directly from ``spider_envelop`` config."""

    condition_names: Tuple[str, ...]
    low: Tuple[float, ...]
    high: Tuple[float, ...]
    morphology_prior_mode: str
    morphology_prior_weights: Mapping[str, Mapping[str, float]]
    morphology_middle_front_follow_weight: float

    def bounds(self, device: torch.device) -> Tuple[Tensor, Tensor]:
        return (
            torch.tensor(self.low, dtype=torch.float32, device=device),
            torch.tensor(self.high, dtype=torch.float32, device=device),
        )


def _find_class(node: ast.ClassDef | ast.Module, name: str) -> ast.ClassDef:
    for child in node.body:
        if isinstance(child, ast.ClassDef) and child.name == name:
            return child
    raise KeyError(f"Class {name!r} not found")


def _literal_assignments(node: ast.ClassDef) -> Dict[str, object]:
    values: Dict[str, object] = {}
    for item in node.body:
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
            try:
                values[item.targets[0].id] = ast.literal_eval(item.value)
            except (ValueError, SyntaxError):
                continue
    return values


def load_envelope_condition_spec(
    config_path: Path | str = DEFAULT_ENV_CONFIG_PATH,
) -> EnvelopeConditionSpec:
    """Read condition order/ranges and morphology settings without importing Isaac Gym."""
    tree = ast.parse(Path(config_path).read_text(encoding="utf-8"))
    config_class = _find_class(tree, "El4090EnvelopCfg")
    commands_class = _find_class(config_class, "commands")
    ranges_class = _find_class(commands_class, "ranges")
    commands = _literal_assignments(commands_class)
    ranges = _literal_assignments(ranges_class)
    condition_names = tuple(cast(Sequence[str], commands["condition_names"]))
    missing = [name for name in condition_names if name not in ranges]
    if missing:
        raise ValueError(f"Missing condition ranges in {config_path}: {missing}")
    return EnvelopeConditionSpec(
        condition_names=condition_names,
        low=tuple(float(cast(Sequence[float], ranges[name])[0]) for name in condition_names),
        high=tuple(float(cast(Sequence[float], ranges[name])[1]) for name in condition_names),
        morphology_prior_mode=str(commands.get("morphology_prior_mode", "directional_ratio")),
        morphology_prior_weights=cast(
            Mapping[str, Mapping[str, float]],
            commands.get(
                "morphology_prior_weights",
                {
                    "front": {"lateral": 0.5, "longitudinal": 0.5},
                    "middle": {"lateral": 1.0},
                    "back": {"lateral": 0.5, "longitudinal": 0.5},
                },
            ),
        ),
        morphology_middle_front_follow_weight=float(
            cast(Any, commands.get("morphology_middle_front_follow_weight", 0.0))
        ),
    )


def apply_env_morphology_priors(condition: Tensor, spec: EnvelopeConditionSpec) -> Tensor:
    """Apply the same prior derivation used by ``spider_envelop._set_morphology_prior_from_envelope``."""
    if condition.shape[-1] != len(spec.condition_names):
        raise ValueError(
            f"Expected condition width {len(spec.condition_names)}, got {condition.shape[-1]}"
        )
    condition = condition.clone()
    index = {name: idx for idx, name in enumerate(spec.condition_names)}
    low = condition.new_tensor(spec.low)
    high = condition.new_tensor(spec.high)

    def normalized(name: str) -> Tensor:
        value = condition[..., index[name]]
        value_low = low[index[name]]
        value_high = high[index[name]]
        if name == "backward_limit":
            value = -value
            value_low = -high[index[name]]
            value_high = -low[index[name]]
        return ((value - value_low) / (value_high - value_low).clamp_min(1e-6)).clamp(0.0, 1.0)

    if spec.morphology_prior_mode == "weighted_sum":
        prior_specs = {
            "morphology_front_prior": {"front_width": -0.5, "forward_limit": 0.5},
            "morphology_middle_prior": {"middle_width": -1.0},
            "morphology_back_prior": {"back_width": -0.5, "backward_limit": 0.5},
        }
        for prior_name, prior_weights in prior_specs.items():
            prior = torch.zeros_like(condition[..., 0])
            weight_sum = 0.0
            for name, weight in prior_weights.items():
                value = normalized(name)
                if weight < 0.0:
                    value = 1.0 - value
                prior = prior + abs(weight) * value
                weight_sum += abs(weight)
            condition[..., index[prior_name]] = (prior / max(weight_sum, 1e-6)).clamp(0.0, 1.0)
        return condition
    if spec.morphology_prior_mode != "directional_ratio":
        raise ValueError(f"Unknown morphology_prior_mode={spec.morphology_prior_mode!r}")

    morphology_weights = spec.morphology_prior_weights
    front_lateral = float(morphology_weights["front"].get("lateral", 0.5)) * normalized("front_width")
    front_longitudinal = float(morphology_weights["front"].get("longitudinal", 0.5)) * normalized("forward_limit")
    back_lateral = float(morphology_weights["back"].get("lateral", 0.5)) * normalized("back_width")
    back_longitudinal = float(morphology_weights["back"].get("longitudinal", 0.5)) * normalized("backward_limit")
    eps = 1e-6
    front_prior = front_longitudinal / (front_longitudinal + front_lateral).clamp_min(eps)
    back_prior = back_longitudinal / (back_longitudinal + back_lateral).clamp_min(eps)
    middle_prior = 1.0 - normalized("middle_width")
    if float(morphology_weights["middle"].get("lateral", 1.0)) <= 0.0:
        middle_prior = torch.full_like(middle_prior, 0.5)
    follow = min(max(spec.morphology_middle_front_follow_weight, 0.0), 1.0)
    middle_prior = follow * front_prior + (1.0 - follow) * middle_prior

    condition[..., index["morphology_front_prior"]] = front_prior.clamp(0.0, 1.0)
    condition[..., index["morphology_middle_prior"]] = middle_prior.clamp(0.0, 1.0)
    condition[..., index["morphology_back_prior"]] = back_prior.clamp(0.0, 1.0)
    return condition


@dataclass(frozen=True)
class HaaRangeConfig:
    """Geometry and training-independent limits used by all estimators."""

    condition_names: Tuple[str, ...] = DEFAULT_CONDITION_NAMES
    leg_names: Tuple[str, ...] = DEFAULT_LEG_NAMES
    joint_lower: float = -3.0
    joint_upper: float = 3.0
    leg_reach: float = 0.55
    front_hip_offset: float = 0.10
    middle_hip_offset: float = 0.20
    back_hip_offset: float = 0.10
    spider_swing_limit: float = 1.05
    mammal_swing_limit: float = 0.45
    minimum_half_range: float = 0.05
    spider_haa: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mammal_haa: Tuple[float, ...] = (-1.308, 1.308, 1.308, -1.308, 1.308, 1.308)

    def __post_init__(self) -> None:
        if len(self.leg_names) != len(self.spider_haa) or len(self.leg_names) != len(self.mammal_haa):
            raise ValueError("leg_names, spider_haa and mammal_haa must have equal lengths")
        missing = {
            "front_width",
            "middle_width",
            "back_width",
            "morphology_front_prior",
            "morphology_middle_prior",
            "morphology_back_prior",
        }.difference(self.condition_names)
        if missing:
            raise ValueError(f"condition_names is missing required fields: {sorted(missing)}")
        if self.joint_lower >= self.joint_upper:
            raise ValueError("joint_lower must be smaller than joint_upper")
        if self.leg_reach <= 0.0:
            raise ValueError("leg_reach must be positive")


class AnalyticHaaRangeEstimator:
    """Closed-form, fully vectorized HAA range estimator."""

    def __init__(self, config: HaaRangeConfig = HaaRangeConfig()) -> None:
        self.config = config
        self._condition_index = {name: index for index, name in enumerate(config.condition_names)}

    def _leg_inputs(self, condition: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        widths: Dict[str, Tensor] = {
            "F": condition[..., self._condition_index["front_width"]],
            "M": condition[..., self._condition_index["middle_width"]],
            "B": condition[..., self._condition_index["back_width"]],
        }
        priors: Dict[str, Tensor] = {
            "F": condition[..., self._condition_index["morphology_front_prior"]],
            "M": condition[..., self._condition_index["morphology_middle_prior"]],
            "B": condition[..., self._condition_index["morphology_back_prior"]],
        }
        offsets = {
            "F": self.config.front_hip_offset,
            "M": self.config.middle_hip_offset,
            "B": self.config.back_hip_offset,
        }
        leg_widths = torch.stack([widths[name[1]] for name in self.config.leg_names], dim=-1)
        leg_priors = torch.stack([priors[name[1]] for name in self.config.leg_names], dim=-1).clamp(0.0, 1.0)
        leg_offsets = condition.new_tensor([offsets[name[1]] for name in self.config.leg_names])
        return leg_widths, leg_priors, leg_offsets

    def centers_and_half_ranges(self, condition: Tensor) -> Tuple[Tensor, Tensor]:
        """Return morphology centers and feasible half ranges, both ``[..., 6]``."""
        if condition.shape[-1] != len(self.config.condition_names):
            raise ValueError(
                f"Expected condition width {len(self.config.condition_names)}, got {condition.shape[-1]}"
            )
        widths, priors, hip_offsets = self._leg_inputs(condition)
        spider = condition.new_tensor(self.config.spider_haa)
        mammal = condition.new_tensor(self.config.mammal_haa)
        centers = spider + priors * (mammal - spider)

        lateral_ratio = ((widths - hip_offsets) / self.config.leg_reach).clamp(0.0, 1.0)
        geometric_limit = torch.asin(lateral_ratio)
        morphology_limit = (
            self.config.spider_swing_limit
            + priors * (self.config.mammal_swing_limit - self.config.spider_swing_limit)
        )
        joint_limit = torch.minimum(
            centers - self.config.joint_lower,
            self.config.joint_upper - centers,
        ).clamp_min(0.0)
        half_range = torch.minimum(torch.minimum(geometric_limit, morphology_limit), joint_limit)
        half_range = torch.minimum(
            half_range.clamp_min(self.config.minimum_half_range),
            joint_limit,
        )
        return centers, half_range

    def __call__(self, condition: Tensor) -> Tensor:
        centers, half_range = self.centers_and_half_ranges(condition)
        return torch.stack((centers - half_range, centers + half_range), dim=-1)


class MonteCarloHaaRangeEstimator(AnalyticHaaRangeEstimator):
    """Sampling estimator useful for validating or generating network labels."""

    def __init__(
        self,
        config: HaaRangeConfig = HaaRangeConfig(),
        num_samples: int = 2048,
        quantile: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config)
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2")
        if not 0.0 <= quantile < 0.5:
            raise ValueError("quantile must be in [0, 0.5)")
        self.num_samples = num_samples
        self.quantile = quantile
        self.seed = seed

    def __call__(self, condition: Tensor) -> Tensor:
        centers, half_range = self.centers_and_half_ranges(condition)
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=condition.device)
            generator.manual_seed(self.seed)
        samples = torch.rand(
            (*centers.shape, self.num_samples),
            dtype=condition.dtype,
            device=condition.device,
            generator=generator,
        )
        candidates = self.config.joint_lower + samples * (self.config.joint_upper - self.config.joint_lower)
        feasible = torch.abs(candidates - centers.unsqueeze(-1)) <= half_range.unsqueeze(-1)

        high_fill = torch.full_like(candidates, self.config.joint_upper)
        feasible_sorted = torch.sort(torch.where(feasible, candidates, high_fill), dim=-1).values
        feasible_count = feasible.sum(dim=-1)
        if self.quantile == 0.0:
            low_index = torch.zeros_like(feasible_count)
            high_index = (feasible_count - 1).clamp_min(0)
        else:
            count_minus_one = (feasible_count - 1).clamp_min(0).to(condition.dtype)
            low_index = torch.floor(self.quantile * count_minus_one).long()
            high_index = torch.ceil((1.0 - self.quantile) * count_minus_one).long()
        low = torch.gather(feasible_sorted, -1, low_index.unsqueeze(-1)).squeeze(-1)
        high = torch.gather(feasible_sorted, -1, high_index.unsqueeze(-1)).squeeze(-1)

        no_feasible = ~feasible.any(dim=-1)
        low = torch.where(no_feasible, centers - half_range, low)
        high = torch.where(no_feasible, centers + half_range, high)
        return torch.stack((low, high), dim=-1)


class HaaRangeNetwork(nn.Module):
    """MLP distillation model that always emits ordered, joint-bounded ranges."""

    def __init__(
        self,
        config: HaaRangeConfig = HaaRangeConfig(),
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_dims = tuple(hidden_dims)
        layers: List[nn.Module] = []
        input_dim = len(config.condition_names)
        for hidden_dim in self.hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ELU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 2 * len(config.leg_names)))
        self.model = nn.Sequential(*layers)

    def forward(self, condition: Tensor) -> Tensor:
        raw = self.model(condition).reshape(*condition.shape[:-1], len(self.config.leg_names), 2)
        center = self.config.joint_lower + torch.sigmoid(raw[..., 0]) * (
            self.config.joint_upper - self.config.joint_lower
        )
        max_half_range = torch.minimum(
            center - self.config.joint_lower,
            self.config.joint_upper - center,
        )
        half_range = torch.sigmoid(raw[..., 1]) * max_half_range
        return torch.stack((center - half_range, center + half_range), dim=-1)

    def fit_estimator(
        self,
        condition_low: Tensor,
        condition_high: Tensor,
        estimator: AnalyticHaaRangeEstimator,
        *,
        steps: int = 2000,
        batch_size: int = 1024,
        learning_rate: float = 3e-4,
        condition_transform: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> float:
        """Train from randomly sampled conditions and return the final MSE."""
        if steps < 1 or batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        final_loss = 0.0
        self.train()
        for _ in range(steps):
            condition = condition_low + torch.rand(
                batch_size,
                condition_low.numel(),
                device=condition_low.device,
                dtype=condition_low.dtype,
            ) * (condition_high - condition_low)
            if condition_transform is not None:
                condition = condition_transform(condition)
            with torch.no_grad():
                target = estimator(condition)
            prediction = self(condition)
            loss = torch.mean(torch.square(prediction - target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
        return final_loss

    def save_checkpoint(self, path: Path | str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": asdict(self.config),
                "hidden_dims": self.hidden_dims,
            },
            Path(path),
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str,
        *,
        device: torch.device | str = "cpu",
    ) -> "HaaRangeNetwork":
        try:
            checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
        except TypeError:  # PyTorch < 2.0 has no weights_only argument.
            checkpoint = torch.load(Path(path), map_location=device)
        config_dict = checkpoint["config"]
        for key in ("condition_names", "leg_names", "spider_haa", "mammal_haa"):
            config_dict[key] = tuple(config_dict[key])
        network = cls(
            HaaRangeConfig(**config_dict),
            hidden_dims=tuple(checkpoint.get("hidden_dims", (128, 128))),
        ).to(device)
        network.load_state_dict(checkpoint["state_dict"])
        network.eval()
        return network


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train an HAA swing-range distillation network.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--labels", choices=("analytic", "monte_carlo"), default="analytic")
    parser.add_argument("--samples", type=int, default=2048, help="Monte-Carlo samples per condition.")
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_CONFIG_PATH)
    args = parser.parse_args(list(argv) if argv is not None else None)

    device = torch.device(args.device)
    condition_spec = load_envelope_condition_spec(args.config)
    config = HaaRangeConfig(condition_names=condition_spec.condition_names)
    estimator: AnalyticHaaRangeEstimator
    if args.labels == "monte_carlo":
        estimator = MonteCarloHaaRangeEstimator(config, num_samples=args.samples)
    else:
        estimator = AnalyticHaaRangeEstimator(config)
    network = HaaRangeNetwork(config).to(device)
    low, high = condition_spec.bounds(device)
    loss = network.fit_estimator(
        low,
        high,
        estimator,
        steps=args.steps,
        batch_size=args.batch_size,
        condition_transform=lambda value: apply_env_morphology_priors(value, condition_spec),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    network.save_checkpoint(args.output)
    print(f"saved={args.output} final_mse={loss:.8f}")


if __name__ == "__main__":
    main()
