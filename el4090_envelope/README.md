# el4090_envelope

`el4090_envelope` is the standalone mathematical implementation of the EL4090
occupied-body support-envelope model. It provides batched Torch forward
kinematics, capsule support functions, finite-normal envelopes, deterministic
range export, and reference-pinned or existential rejection projections. The
installable core has no Isaac Gym, RL environment, web framework, or
`agent_team` dependency; optional repository examples are kept outside `src/`.

```mermaid
flowchart LR
    urdf["EL4090 URDF and joint pose"] --> core["el4090_envelope core"]
    core --> support["Occupied support and feasible envelope"]
    core --> ranges["Joint ranges and rejection intervals"]
    support --> viewers["Web and Isaac Gym examples"]
    ranges --> consumers["Planning and RL integrations"]
```

## Layout

- `src/el4090_envelope/model.py`: kinematics, capsules, support/envelope and rejection math.
- `src/el4090_envelope/geometry.py`: reusable half-space and arc geometry.
- `tests/`: deterministic unit, regression, edge-case, and serialization checks.
- `examples/web_viewer/`: self-contained ENV-VIS-015 interactive viewer.
- `examples/isaac_gym/`: original kinematic, LiDAR, and slider viewers.
- `docs/`: mathematics, implementation/API, and migration notes.

## Use from a source checkout

From this directory, use the project's `src` tree explicitly:

```bash
PYTHONPATH=src python -c "import el4090_envelope; print(el4090_envelope.__file__)"
PYTHONPATH=src python -m unittest discover -s tests -v
```

Basic model construction:

```python
from pathlib import Path
import torch
from el4090_envelope import (
    BatchedUrdfKinematics, capsule_support, default_el4090_capsules,
    load_urdf_joints, support_directions,
)

kinematics = BatchedUrdfKinematics(load_urdf_joints(Path("robot.urdf")))
directions = support_directions(32)
q = torch.zeros(1, 18)
occupied_support = capsule_support(
    kinematics, q, default_el4090_capsules(), directions,
)
```

Continue with [docs/usage.md](docs/usage.md) for detailed API workflows,
[docs/math.md](docs/math.md) for the model and approximation contract,
[docs/implementation.md](docs/implementation.md) for architecture/API ownership,
[docs/isaac_gym_examples.md](docs/isaac_gym_examples.md) for the original viewer
examples, and [docs/migration.md](docs/migration.md) for legacy imports.

## Web viewer

```bash
PYTHONPATH=src python examples/web_viewer/envelope_server.py --port 8766
```

Open `http://127.0.0.1:8766/`. The server defaults to the sibling repository's
EL4090 URDF; set `EL4090_URDF=/absolute/path/to/el_4090.urdf` elsewhere.
