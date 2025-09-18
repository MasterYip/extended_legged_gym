# Extended Isaac Gym Environments for Legged Robots

<div align="center">
  <img src="doc/teaser1.png" alt="Terrain Navigation" width="42%" style="margin-right: 0%"/>
  <img src="doc/teaser2.png" alt="Multi-Robot Environment" width="45%"/>
</div>

> [!WARNING]
> This repository is still under development. Documentation is incomplete and the code may contain bugs.

This repository extends the original [legged_gym](https://github.com/leggedrobotics/legged_gym).
And is used as an submodule in [PegasusFlow](https://github.com/MasterYip/PegasusFlow)

## Newly Added Features

- **`rsl_rl` 3.3.0 support**: Update support from rsl_rl 1.0.2 to 3.3.0.
- **Nvidia Warp SDF & Raycasting**: Integration of Nvidia Warp SDF, raycasting and depth camera for enhanced environment interaction.
- **Main-Rollout Environment Architecture**: Implementation of a main-rollout architecture for sampling-based methods.

<div class="columns is-centered has-text-centered is-vcentered">
    <div class="column is-fullwidth is-centered">
        <video id="method_video" autoplay controls muted loop playsinline width="70%">
            <source src="doc/anymal_rollout.mp4" type="video/mp4">
        </video>
    </div>
</div>

https://github.com/user-attachments/assets/f9a9bcac-ec0e-4ffe-bc07-01bdd7ab75f7

- **Confined Terrain Generation & OBJ Terrain Support**: Added confined terrain generation and support for OBJ terrains. To generate OBJ terrains, you can refer to [leggedrobotics/terrain-generator](https://github.com/leggedrobotics/terrain-generator), [MasterYip/blender_robotic_utils](https://github.com/MasterYip/blender_robotic_utils).
- **Miscellaneous Enhancements**:
  - gym_visualizer integration
  - benchmarking tools
  - etc.
