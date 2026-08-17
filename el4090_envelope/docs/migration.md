# Migration and compatibility

## Source map

| Previous source | New owner |
|---|---|
| `legged_gym.utils.envelop.kinematic_envelope` | `el4090_envelope.model` / package root |
| `legged_gym.utils.envelop.gym_envelope_geometry` | `el4090_envelope.geometry` |
| ENV-VIS-015 task website | `examples/web_viewer` |
| legacy tests | `tests/test_model.py`, `tests/test_geometry.py` |
| benchmark, Isaac viewers, RL adapters | remain outside package core |

## Base-tree inventory and disposition

| Base path under `utils/envelop` | Classification and disposition |
|---|---|
| `kinematic_envelope.py` | Reusable model math; moved byte-for-byte to `model.py`, old path is a facade. |
| `gym_envelope_geometry.py` | Isaac-independent planar/kinematic geometry; moved to `geometry.py` with only its import made package-relative. |
| `lidar_free_envelope.py`, `legacy_slider_envelope.py` | Executable/demo integration; retained outside core and verified through legacy tests. |
| `draw_envelope.py`, `morphology_prior.py` | Plotting/project-configuration examples; retained in the simulator tree. |
| `benchmark_kinematic_envelope.py`, `figures/*.json` | Benchmark harness/evidence; excluded from the distribution core. |
| `network/` | Policy condition estimator and training/plotting integration, not occupied-support math; retained unchanged. |
| `pd_control.py` | Unrelated control utility; retained unchanged. |
| `*.png`, `figures/` | Generated media/evidence; not copied into the package. |
| `KINEMATIC_ENVELOPE.md`, legacy `README.md` | Source documentation consolidated into this package's README and `docs/`. |

No weights, checkpoints, rendered images, raw benchmark output, URDF, RL
configuration, or simulator code are included in the standalone project.

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
