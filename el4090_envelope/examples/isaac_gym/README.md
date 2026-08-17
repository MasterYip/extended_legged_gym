# Isaac Gym examples

These are the original EL4090 kinematic, LiDAR free-envelope, and legacy-slider
viewers. They remain separate examples; they are not folded into the web viewer
or package core.

Run from the `el4090_envelope` directory with an Isaac Gym environment and the
sibling `legged_gym/resources` tree available:

```bash
PYTHONPATH=src python examples/isaac_gym/visualize_kinematic_envelope_gym.py --compute_only
PYTHONPATH=src python examples/isaac_gym/visualize_lidar_free_envelope_gym.py --compute_only
PYTHONPATH=src python examples/isaac_gym/visualize_legacy_slider_envelope_gym.py --compute_only
```

See [../../docs/isaac_gym_examples.md](../../docs/isaac_gym_examples.md) for
controls, bounded runs, evidence capture, mathematical context, and
troubleshooting.
