# Migration and compatibility

## Source map

| Previous source | New owner |
|---|---|
| `legged_gym.utils.envelop.kinematic_envelope` | `el4090_envelope.model` / package root |
| `legged_gym.utils.envelop.gym_envelope_geometry` | `el4090_envelope.geometry` |
| ENV-VIS-015 task website | `examples/web_viewer` |
| legacy core tests | replaced by `tests/test_model.py`, `tests/test_geometry.py` |
| legacy core benchmark | removed after package verification |
| original Isaac viewers and support | moved to `examples/isaac_gym` |

## Base-tree inventory and disposition

| Base path under `utils/envelop` | Classification and disposition |
|---|---|
| `kinematic_envelope.py` | Reusable model math; moved byte-for-byte to `model.py`, old path is a facade. |
| `gym_envelope_geometry.py` | Isaac-independent planar/kinematic geometry; moved to `geometry.py` with only its import made package-relative. |
| `lidar_free_envelope.py`, `legacy_slider_envelope.py`, `pd_control.py` | Example support; moved to `examples/isaac_gym` and verified by package tests. |
| `draw_envelope.py`, `morphology_prior.py` | Shared rendering/configuration helpers for the retained HAA-network diagnostic; standalone generated images removed. |
| `benchmark_kinematic_envelope.py`, `figures/*.json` | Superseded core benchmark and generated evidence; removed from the simulator tree. |
| `network/` | Policy condition estimator and training/plotting integration, not occupied-support math; retained unchanged. |
| top-level `*.png`, `figures/` | Generated legacy media/evidence; removed rather than copied into the package. |
| `KINEMATIC_ENVELOPE.md` | Superseded source documentation; consolidated into this package's README and `docs/`, then removed. |
| legacy `README.md` | Replaced by a concise integration ownership map. |
| three `scripts/visualize_*_envelope_gym.py` examples | Moved unchanged in purpose to `examples/isaac_gym`; imports/path ownership adapted to the package. |
| core and example tests | Replaced or moved into package `tests/`; no duplicate envelope tests remain in `legged_gym`. |
| `doc/envelope_visualization.md` | Migrated to `docs/isaac_gym_examples.md`, with package-owned command paths. |

No weights, checkpoints, rendered images, raw benchmark output, URDF, or RL
configuration are included. Optional simulator viewers are repository examples
outside `src/` and are intentionally excluded from the core wheel.

The two legacy module files are compatibility facades. They locate this
checkout's `el4090_envelope/src`, import the package modules, and re-export the
historical namespace. There is one mathematical implementation. Existing
path-based scripts therefore keep working while new code should use:

```python
from el4090_envelope import capsule_support, haa_rejection_ranges
from el4090_envelope.geometry import support_polygon
```

Public signatures and numeric behavior at base commit `8b3f0734` are retained,
including ENV-KINE-016 `pinned`/`fold`/`none` semantics and ENV-BENCH-017 tuned
fold defaults. No URDF, RL observation/action/reward, environment configuration,
checkpoint, or simulator behavior changes are part of this migration.

The viewer no longer searches `agent_team/`. From the repository worktree it
finds `el4090_envelope/src` and the sibling `legged_gym/resources` tree. For a
separate installation, install the distribution and set `EL4090_URDF`.
