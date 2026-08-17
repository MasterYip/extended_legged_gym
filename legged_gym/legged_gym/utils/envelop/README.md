# EL4090 envelope integration

The reusable occupied-envelope mathematics is owned by the standalone
`el4090_envelope` distribution at the repository root. New code should import
its public API directly. `kinematic_envelope.py` and
`gym_envelope_geometry.py` remain here only as compatibility facades for
existing `legged_gym` imports.

This directory retains simulator/runtime integration that does not belong in
the package distribution:

- `network/`: the HAA estimator, training, and diagnostic visualization path.
- `draw_envelope.py` and `morphology_prior.py`: shared rendering/configuration
  helpers used by the HAA-network diagnostic viewer.

Package mathematics, API documentation, deterministic unit tests, and the
web/Isaac examples live under `el4090_envelope/`. See
`el4090_envelope/docs/migration.md` for the ownership map.
