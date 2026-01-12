import torch
import numpy as np
from typing import List, Optional, Dict, Any
from isaacgym import gymtorch, gymapi
import time
import os
from PIL import Image


class RecordReplayMixin:
    """
    A mixin class for legged robot environments that provides functionality 
    to record and replay simulation steps.

    This allows recording trajectories and replaying them for visualization,
    analysis, or debugging purposes.
    """

    def record_replay_mixin_init(self):
        """
        Initialize the recording variables.

        Note: This should be called after the environment's __init__ method.
        """
        # Initialize record/replay variables
        self.is_recording = False
        self.is_replaying = False
        self.recorded_root_states = []
        self.recorded_dof_states = []
        self.replay_index = 0
        self.max_replay_length = 0
        self.record_render_frames = False
        self.render_frames = []

        # Snapshot variables
        self.snapshot_counter = 0

    # === Recording === #

    def start_recording(self, max_steps: int = 10000, record_render: bool = False) -> None:
        """
        Start recording the simulation states.

        Args:
            max_steps: Maximum number of steps to record
            record_render: Whether to also record rendered frames
        """
        self.is_recording = True
        self.is_replaying = False
        self.recorded_root_states = []
        self.recorded_dof_states = []
        self.record_render_frames = record_render
        self.render_frames = []
        self.max_record_length = max_steps
        print(f"Recording started (max {max_steps} steps)")

    def stop_recording(self) -> None:
        """Stop recording and convert lists to tensors."""
        self.is_recording = False
        if len(self.recorded_root_states) > 0:
            self.recorded_root_states = torch.stack(self.recorded_root_states)
            self.recorded_dof_states = torch.stack(self.recorded_dof_states)
            self.max_replay_length = len(self.recorded_root_states)
            print(f"Recording stopped. Recorded {self.max_replay_length} steps.")
        else:
            print("Recording stopped. No steps were recorded.")

    def record_step(self) -> None:
        """Record the current simulation state."""
        if not self.is_recording or len(self.recorded_root_states) >= self.max_record_length:
            return

        # Record root states and DOF states by making a copy
        self.recorded_root_states.append(self.root_states.clone())
        self.recorded_dof_states.append(self.dof_state.clone())

        # Optionally record rendered frames if requested
        if self.record_render_frames and hasattr(self, "headless") and not self.headless:
            # This would require implementing a method to capture rendered frames
            pass

    # === Viewer Snapshot === #

    def save_viewer_snapshot(self, output_dir: str = "./snapshots", 
                             filename_prefix: str = "snapshot") -> Optional[str]:
        """
        Save a snapshot from the viewer using write_viewer_image_to_file.

        Args:
            output_dir: Directory to save snapshots
            filename_prefix: Prefix for saved files

        Returns:
            Path to saved file, or None if failed
        """
        if getattr(self, "headless", True) or getattr(self, "viewer", None) is None:
            print("Error: No viewer available (headless mode). Cannot save viewer snapshot.")
            return None

        if not hasattr(self, 'gym'):
            print("Error: Gym not available.")
            return None

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Generate filename
        filename = os.path.join(output_dir, f"{filename_prefix}_{self.snapshot_counter:06d}.png")

        try:
            # Use gym's built-in viewer image saving function
            self.gym.write_viewer_image_to_file(self.viewer, filename)
            self.snapshot_counter += 1
            print(f"Viewer snapshot saved: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving viewer snapshot: {e}")
            return None

    # === Replay === #

    def start_replay(self, loop: bool = True) -> None:
        """
        Start replaying the recorded simulation steps.

        Args:
            loop: Whether to loop the replay once it reaches the end
        """
        if len(self.recorded_root_states) == 0 or not isinstance(self.recorded_root_states, torch.Tensor):
            print("No recorded data available for replay")
            time.sleep(1)
            return

        self.is_replaying = True
        self.is_recording = False
        self.replay_index = 0
        self.replay_loop = loop
        print(f"Replay started with {self.max_replay_length} steps")

    def stop_replay(self) -> None:
        """Stop the replay."""
        self.is_replaying = False
        print("Replay stopped")

    def replay(self) -> bool:
        """
        Replay one recorded step by setting the simulation state.

        Returns:
            bool: True if the replay continues, False if replay has finished
        """
        if not self.is_replaying or not hasattr(self, "recorded_root_states") or len(self.recorded_root_states) == 0:
            return False

        if self.replay_index >= self.max_replay_length:
            if self.replay_loop:
                self.replay_index = 0
            else:
                self.is_replaying = False
                return False

        # Set the root states and DOF states from the recording
        self.gym.set_actor_root_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.recorded_root_states[self.replay_index])
        )

        self.gym.set_dof_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.recorded_dof_states[self.replay_index])
        )

        # Simulate the environment for one step
        self.gym.simulate(self.sim)

        # Refresh the state tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        if self.viewer and self.debug_viz:
            self._draw_debug_vis()
        # Render the scene if in GUI mode
        if hasattr(self, "render") and not getattr(self, "headless", True):
            self.render()

        self.replay_index += 1
        return True

    def replay_blocking(self, num_steps: Optional[int] = None, render: bool = True) -> bool:
        """
        Replay recorded simulation steps in blocking mode until completion or for specified number of steps.

        This method will continuously replay steps until either:
        1. The entire recording has been replayed (and loop=False)
        2. The specified number of steps have been replayed
        3. The replay has been manually stopped

        Args:
            num_steps: Maximum number of steps to replay (None for all available steps)
            render: Whether to render each step during replay

        Returns:
            bool: True if replay completed successfully, False if no data or replay was stopped
        """
        if not hasattr(self, "recorded_root_states") or len(self.recorded_root_states) == 0:
            print("No recorded data available for replay")
            time.sleep(1)
            return False

        # Start replay if not already replaying
        if not self.is_replaying:
            self.start_replay(loop=(num_steps is None))

        # Determine how many steps to replay
        steps_to_replay = num_steps if num_steps is not None else self.max_replay_length
        steps_replayed = 0

        print(f"Starting blocking replay for {steps_to_replay} steps")

        # Main replay loop
        while self.is_replaying and steps_replayed < steps_to_replay:
            # Replay one step
            if not self.replay():
                break

            steps_replayed += 1
            time.sleep(self.dt)  # Default dt

        # Return success status
        success = steps_replayed == steps_to_replay
        if success:
            print(f"Replay completed: {steps_replayed} steps replayed")
        else:
            print(f"Replay stopped after {steps_replayed} steps")

        return success

    # === Other === #

    def get_recording_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the recorded data.

        Returns:
            Dict containing recording statistics
        """
        if not hasattr(self, "recorded_root_states") or len(self.recorded_root_states) == 0:
            return {"status": "No recording available"}

        num_steps = self.max_replay_length
        duration = num_steps * self.dt if hasattr(self, "dt") else num_steps * 0.02  # Default dt

        return {
            "status": "Recording available",
            "num_steps": num_steps,
            "duration_seconds": duration,
            "num_environments": (self.recorded_root_states[0].shape[0]
                                 if isinstance(self.recorded_root_states, list)
                                 else self.recorded_root_states.shape[1])
        }

    def export_recording(self, file_path: str) -> bool:
        """
        Export the recorded data to a file.

        Args:
            file_path: Path to save the recording

        Returns:
            bool: True if export was successful
        """
        if not hasattr(self, "recorded_root_states") or len(self.recorded_root_states) == 0:
            print("No recording available to export")
            return False

        try:
            torch.save({
                "root_states": self.recorded_root_states,
                "dof_states": self.recorded_dof_states,
                "dt": self.dt if hasattr(self, "dt") else 0.02,
                "num_envs": self.num_envs if hasattr(self, "num_envs") else 1
            }, file_path)
            print(f"Recording exported to {file_path}")
            return True
        except Exception as e:
            print(f"Failed to export recording: {e}")
            return False

    def import_recording(self, file_path: str) -> bool:
        """
        Import a recording from a file.

        Args:
            file_path: Path to the recording file

        Returns:
            bool: True if import was successful
        """
        try:
            data = torch.load(file_path)
            self.recorded_root_states = data["root_states"]
            self.recorded_dof_states = data["dof_states"]
            self.max_replay_length = len(self.recorded_root_states)
            print(f"Imported recording with {self.max_replay_length} steps")
            return True
        except Exception as e:
            print(f"Failed to import recording: {e}")
            return False
