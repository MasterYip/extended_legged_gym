# ElSpider LiDAR Confined Space Navigation Task
# 基于激光雷达的六足机器人受限空间强化学习避障运动控制

## Overview

This module implements a LiDAR-enabled hexapod robot (ElSpider) training environment for obstacle avoidance and navigation in confined spaces. The robot uses simulated LiDAR sensors to perceive its environment and learn collision avoidance behaviors through reinforcement learning.

## Architecture

### Environment Class Hierarchy
```
LeggedRobot (base)
    └── ElSpider (hexapod robot base)
        └── ElSpiderLidar (LiDAR-enabled version)
```

### Key Components

1. **ElSpiderLidar** (`elspider_lidar.py`)
   - Extends ElSpider with LiDAR sensor integration
   - Uses OmniPerception's LidarSensor for ray-casting
   - Processes point cloud data into observation vectors

2. **Configuration** (`elspider_lidar_confined_config.py`)
   - Defines LiDAR parameters (FOV, resolution, range)
   - Configures confined terrain types
   - Sets up reward functions for obstacle avoidance

3. **Terrain** (from `terrain_confine.py`)
   - `tunnel_terrain`: Tunnel passages with low ceilings
   - `barrier_terrain`: Square barriers around spawn area
   - `timber_piles_terrain`: Random pillar obstacles
   - `confined_gap_terrain`: Gaps and platforms
   - `column_obstacles_terrain`: Vertical columns
   - `wall_with_gap_terrain`: Walls with navigable openings

## Usage

### Training

```bash
cd extended_legged_gym/legged_gym

# Train on mixed confined terrains
python legged_gym/scripts/train.py --task=elspider_lidar_confined --headless

# Train on timber pile terrain only
python legged_gym/scripts/train.py --task=elspider_lidar_timber_pile --headless

# Train on tunnel terrain only  
python legged_gym/scripts/train.py --task=elspider_lidar_tunnel --headless

# Train with simplified LiDAR (faster)
python legged_gym/scripts/train.py --task=elspider_lidar_confined_simple --headless
```

### Playing/Evaluation

```bash
# Play trained policy
python legged_gym/scripts/play.py --task=elspider_lidar_confined --load_run=<run_name>

# With visualization
python legged_gym/scripts/play.py --task=elspider_lidar_confined --num_envs=1
```

## Observation Space

The observation vector consists of:

| Index Range | Dimension | Description |
|-------------|-----------|-------------|
| 0-2 | 3 | Base linear velocity (scaled) |
| 3-5 | 3 | Base angular velocity (scaled) |
| 6-8 | 3 | Projected gravity vector |
| 9-11 | 3 | Velocity commands |
| 12-29 | 18 | DOF positions (relative to default) |
| 30-47 | 18 | DOF velocities |
| 48-65 | 18 | Previous actions |
| 66-252 | 187 | Height measurements (17×11 grid) |
| 253-348 | 96 | LiDAR observations (12×8 bins) |

**Total: 349 observations**

## LiDAR Configuration

Default LiDAR settings:
- **Type**: Simple grid pattern
- **Horizontal FOV**: 360° (-180° to 180°)
- **Vertical FOV**: 45° (-30° to 15°)
- **Horizontal Rays**: 36
- **Vertical Rays**: 10
- **Max Range**: 5.0 meters
- **Update Rate**: 20 Hz

## Reward Structure

### Locomotion Rewards
- `tracking_lin_vel`: Following velocity commands
- `tracking_ang_vel`: Following angular velocity commands
- `feet_air_time`: Maintaining proper gait
- `orientation`: Keeping body upright

### Confined Space Rewards
- `obstacle_avoidance`: +2.0 for maintaining safe distance
- `collision_penalty`: -5.0 for getting too close to obstacles
- `exploration`: +0.5 for moving forward when safe

### Penalties
- `lin_vel_z`: Vertical velocity penalty
- `ang_vel_xy`: Roll/pitch angular velocity penalty
- `action_rate`: Smooth action penalty
- `dof_pos_limits`: Joint limit penalty

## Training Tips

1. **Start with simple terrain**: Use `elspider_lidar_timber_pile` first
2. **Use curriculum**: Enable terrain curriculum for gradual difficulty increase
3. **Monitor LiDAR observations**: Use visualization to debug sensor issues
4. **Tune collision threshold**: Adjust `collision_threshold` based on robot size

## Dependencies

- Isaac Gym (Preview 4+)
- OmniPerception LidarSensor
- rsl_rl (reinforcement learning)
- warp (GPU ray-casting)
- trimesh (mesh processing)

## File Structure

```
elspider_air/
├── elspider.py                    # Base ElSpider class
├── elspider_lidar.py              # LiDAR-enabled ElSpider
├── elspider_lidar_confined_config.py  # Configuration
└── README_LIDAR_CONFINED.md       # This file
```

## Customization

### Adding New Terrain Types

1. Add terrain generation function in `terrain_confine.py`
2. Update `confined_terrain_proportions` in config
3. Modify `TerrainConfined.make_confined_terrain()` to include new type

### Modifying LiDAR Configuration

```python
class lidar:
    sensor_type = "simple_grid"  # or "avia", "mid360"
    horizontal_line_num = 72     # Increase for higher resolution
    vertical_line_num = 16
    max_range = 10.0             # Increase for larger environments
    num_theta_bins = 18          # Observation bins (affects network input)
    num_phi_bins = 12
```

### Adding Custom Rewards

```python
def _reward_my_custom_reward(self):
    # Access LiDAR data
    min_dist = self.min_obstacle_dist
    lidar_obs = self.lidar_obs_buf
    
    # Compute reward
    reward = ...
    return reward
```

## Known Limitations

1. WARP mesh creation is done once at initialization - dynamic obstacles not supported
2. LiDAR update rate affects training speed - use lower rates for faster training
3. Confined terrain generation uses heightfield approach - complex 3D structures limited

## References

- [Extended Legged Gym](../../../README.md)
- [OmniPerception LidarSensor](../../../../OmniPerception/LidarSensor/README.md)
- [RSL-RL](../../../rsl_rl/README.md)
