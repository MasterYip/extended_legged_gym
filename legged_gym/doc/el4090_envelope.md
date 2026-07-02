# EL_4090 Envelope Analyzer

Hexagonal-prism envelope computation and visualization for the EL_4090 hexapod robot.

## Overview

The envelope models the robot's occupied workspace as a **hexagonal prism**:

```text
        Top hexagon (magenta)
       /|    |    |    |    |\
      / |    |    |    |    | \
     /  |    |    |    |    |  \
    *---*----*----*----*----*---*   ← top face = base_height + height_bias
    |   |    |    |    |    |   |
    |   *----*----*----*----*   |   ← 6 vertical edges (yellow)
    |   |    |    |    |    |   |
    *---*----*----*----*----*---*   ← bottom face = max(min_foot_z, min_height)
       \ |    |    |    |    | /
        \|    |    |    |    |/
        Bottom hexagon (cyan)
```

- **2D hexagon** — defined by the 6 foot XY positions, ordered by angle around their centroid.
- **Height** — from the lowest foot (or `min_height`) up to `base_height + height_bias`.

## File Structure

```text
utils/envelope/
    __init__.py
    envelope_calculator.py          # EnvelopeCalculator class

envs/el_4090/envelope/
    __init__.py
    el4090_envelope.py              # EL_4090_Envelope environment
    el4090_envelope_spider_config.py # Config classes
```

## Quick Start

### Environment Setup

All commands below assume you are inside the legged_gym directory with the Isaac Gym conda environment:

```bash
cd /home/user/CodeSpace/Diffusion/PredictiveDiffusionPlanner_Dev/extended_legged_gym/legged_gym
conda activate isaacgym
```

### Code

```python
from legged_gym.envs.el_4090.envelope import (
    EL_4090_Envelope,
    El4090EnvelopeSpiderCfg,
    El4090EnvelopeSpiderCfgPPO,
)

# Create config
cfg = El4090EnvelopeSpiderCfg()

# Create environment (headless=False to see the viewer)
env = EL_4090_Envelope(
    cfg=cfg,
    sim_params=sim_params,
    physics_engine=physics_engine,
    sim_device="cuda:0",
    headless=False,
)

# In the training loop, the envelope is rendered automatically.
# Access envelope data programmatically:
envelope = env.compute_envelope()
print(envelope['bottom_vertices'].shape)  # [num_envs, 6, 3]
print(envelope['top_height'].shape)        # [num_envs]

volume = env.get_envelope_volume()         # [num_envs]
```

---

## Training

Registered task name: `el4090_envelope`.

```bash
# Full training (headless, 4096 envs)
python legged_gym/scripts/train.py --task=el4090_envelope --num_envs=4096 --headless --resume

# Debug / low-env training (512 envs)
python legged_gym/scripts/train.py --task=el4090_envelope --num_envs=512 --headless --resume

# Smoke-test (1 iteration to validate wiring)
python legged_gym/scripts/train.py --task=el4090_envelope --num_envs=64 --headless --max_iterations=1
```

## Evaluation / Play

```bash
# Visual evaluation with envelope rendering (no --headless so viewer opens)
python legged_gym/scripts/play.py --task=el4090_envelope --num_envs=48 --checkpoint=-1 --resume

# Evaluate a specific run
python legged_gym/scripts/play.py --task=el4090_envelope --num_envs=48 --checkpoint=-1 --load_run=<run_name> --resume
```

**Note:** The envelope visualization (hexagonal prism) renders automatically in the viewer.
To disable it, set `enable_envelope_vis = False` in the config or train headless.

---

## Configuration

Add these to your config (already present in `El4090EnvelopeSpiderCfg`):

```python
class env:
    enable_envelope_vis = True       # toggle envelope rendering in viewer
    envelope_vis_interval = 1         # draw every N steps (1 = every step)

class envelope:
    height_bias = 0.30               # [m] offset above base for top face
    min_height = 0.0                  # [m] minimum Z for bottom face
    max_height = None                 # [m] optional cap (None = no cap)
    hexagon_radius_scale = 1.05       # safety margin (>1 adds padding)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `height_bias` | float | `0.30` | Offset above base Z for the top hexagon face |
| `min_height` | float | `0.0` | Floor Z for the bottom face (clamp) |
| `max_height` | float or None | `None` | Ceiling Z for the top face (`None` = no cap) |
| `hexagon_radius_scale` | float | `1.05` | Scale foot radial distances (>1 adds a 5% margin) |

## API Reference

### `EnvelopeCalculator`

Located in `legged_gym.utils.envelope`.

```python
from legged_gym.utils.envelope import EnvelopeCalculator

calc = EnvelopeCalculator(
    height_bias=0.30,
    min_height=0.0,
    max_height=None,
    hexagon_radius_scale=1.05,
)

# Compute envelope from foot positions and base position
# foot_positions: torch.Tensor [num_envs, 6, 3]  (world frame)
# base_pos:        torch.Tensor [num_envs, 3]     (world frame)
result = calc.compute(foot_positions, base_pos)
```

**`compute()` returns:**

| Key | Shape | Description |
| --- | --- | --- |
| `bottom_vertices` | `[num_envs, 6, 3]` | XYZ of bottom hexagon corners |
| `top_vertices` | `[num_envs, 6, 3]` | XYZ of top hexagon corners |
| `bottom_height` | `[num_envs]` | Z value of the bottom face |
| `top_height` | `[num_envs]` | Z value of the top face |
| `hex_center` | `[num_envs, 2]` | (x, y) centre of the hexagon |

**Helper methods:**

```python
# Get edge index pairs for a closed N-gon
edges = calc.get_edges(vertices)         # [(0,1), (1,2), ..., (5,0)]

# Get all three edge sets of the prism at once
bottom_edges, top_edges, vertical_edges = calc.get_prism_edges()
```

### `EL_4090_Envelope`

Located in `legged_gym.envs.el_4090.envelope`. Inherits from `EL_4090`.

```python
envelope = env.compute_envelope()       # dict (cached as env._last_envelope)
volume   = env.get_envelope_volume()    # torch.Tensor [num_envs]
```

**Visualization colors:**

| Element | Color |
| --- | --- |
| Bottom hexagon edges | Cyan `(0, 1, 1)` |
| Top hexagon edges | Magenta `(1, 0, 1)` |
| Vertical pillars | Yellow `(1, 1, 0)` |
| Centre marker | White `(1, 1, 1)` |

## How the Hexagon is Computed

1. Take the 6 foot positions in **world frame** XY: `foot_positions[:, :, :2]`.
2. Compute the centroid: mean of all 6 foot XY positions.
3. Compute the angle of each foot relative to the centroid via `atan2(y, x)`.
4. Sort the 6 feet by angle → this produces an ordered hexagon.
5. Optionally scale each radial vector by `hexagon_radius_scale` to add a safety margin.

This ensures the hexagon is always a simple (non-self-intersecting) polygon that encompasses the current foot positions.

## Class Hierarchy

```text
BaseTask
  └── LeggedRobot (legged_robot.py)
        └── ElSpider (elspider.py)
              └── EL_4090 (spider_nomal/el_4090.py)
                    └── EL_4090_Envelope (envelope/el4090_envelope.py)
```

```text
ElSpiderAirRoughCfg
  └── El4090SpiderCfg (spider_nomal/el4090_spider_config.py)
        └── El4090EnvelopeSpiderCfg (envelope/el4090_envelope_spider_config.py)

ElSpiderAirRoughCfgPPO
  └── El4090SpiderCfgPPO
        └── El4090EnvelopeSpiderCfgPPO
```

## Debugging

Set `cfg.env.debug_mode = True` to enable console output. The `_draw_envelope_info_overlay()` method prints:

```text
============================================================
[Envelope Info] Step 100 | Env 0
------------------------------------------------------------
  Bottom Z:      0.020 m
  Top Z:         0.750 m
  Height:        0.730 m
  Centre XY:     [0.123, -0.005]
  Volume:        0.1523 m^3
============================================================
```

## Extending

To add envelope support to a different robot:

1. Subclass that robot's environment class.
2. Initialize `EnvelopeCalculator` with appropriate parameters.
3. Override `_draw_debug_vis()` to call `compute_envelope()` and render the prism edges via `self.vis.draw_line()`.
4. Create a config class with `envelope` parameters and `env.enable_envelope_vis`.
