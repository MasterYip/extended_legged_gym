# Configuration and Technical Notes

## Terrain Configuration

### Terrain Object Usage

To use the TerrainObj class for terrain creation:

```python
use_terrain_obj = True  # use TerrainObj class to create terrain
```

Then set the terrain file in the `TerrainObj` class.

## Known Issues

### Model File Numbering
- [ ] Some tasks' model files are correctly numbered when resuming, but some are not (e.g., stand_anymal)

### Performance Enhancement
- [ ] Sync main to rollout & cache main
- [ ] Use tensor state instead of dict state for better performance

## Environment Setup

All tasks require the following environment activation:

```bash
conda activate diffuseloco
```

## General Command Structure

### Training Commands
```bash
python legged_gym/scripts/train.py --task=<task_name> --num_envs=<num_environments> --resume --headless
```

### Play/Testing Commands
```bash
python legged_gym/scripts/play.py --task=<task_name> --num_envs=<num_environments> --checkpoint=<checkpoint_number>
```

### Common Parameters
- `--num_envs`: Number of parallel environments (typically 4096-6144 for training, 1-48 for testing)
- `--resume`: Resume training from the last checkpoint
- `--headless`: Run without GUI (for training)
- `--checkpoint`: Specify checkpoint number (-1 for latest)