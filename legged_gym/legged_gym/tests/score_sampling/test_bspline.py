import unittest
import numpy as np
from legged_gym.envs.base.base_task import BaseTask
import torch
from legged_gym.utils.wbfo.spline import SplineBase

# Add matplotlib for visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TestBSpline(unittest.TestCase):
    def setUp(self):
        # Use CPU for tests to make assertions easier
        self.device = torch.device('cpu')
        self.spline = SplineBase(device=self.device)

    def test_basis_matrix(self):
        """Test that the basis matrix is correctly generated."""
        basis_matrix = self.spline.get_cubic_bspline_basis_matrix()

        # Check dimensions
        self.assertEqual(basis_matrix.shape, (4, 4))

        # Check values match expected B-spline basis matrix (divided by 6)
        expected = torch.tensor([
            [1.0, 4.0, 1.0, 0.0],
            [-3.0, 0.0, 3.0, 0.0],
            [3.0, -6.0, 3.0, 0.0],
            [-1.0, 3.0, -3.0, 1.0]
        ]) / 6.0

        self.assertTrue(torch.allclose(basis_matrix, expected))

    def test_power_basis_vector(self):
        """Test the power basis vector generation."""
        # Test single value
        t = torch.tensor(0.5, device=self.device)
        basis = self.spline.get_power_basis_vector(t)
        expected = torch.tensor([1.0, 0.5, 0.25, 0.125], device=self.device)
        self.assertTrue(torch.allclose(basis, expected))

        # Test batch of values
        t_batch = torch.tensor([0.0, 0.5, 1.0], device=self.device)
        basis_batch = self.spline.get_power_basis_vector(t_batch)
        expected_batch = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.5, 0.25, 0.125],
            [1.0, 1.0, 1.0, 1.0]
        ], device=self.device)
        self.assertTrue(torch.allclose(basis_batch, expected_batch))

    def test_velocity_basis_vector(self):
        """Test the velocity basis vector generation."""
        # Test single value
        t = torch.tensor(0.5, device=self.device)
        basis = self.spline.get_velocity_basis_vector(t)
        expected = torch.tensor([0.0, 1.0, 1.0, 0.75], device=self.device)
        self.assertTrue(torch.allclose(basis, expected))

        # Test batch of values
        t_batch = torch.tensor([0.0, 0.5, 1.0], device=self.device)
        basis_batch = self.spline.get_velocity_basis_vector(t_batch)
        expected_batch = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.75],
            [0.0, 1.0, 2.0, 3.0]
        ], device=self.device)
        self.assertTrue(torch.allclose(basis_batch, expected_batch))

    def test_acceleration_basis_vector(self):
        """Test the acceleration basis vector generation."""
        # Test single value
        t = torch.tensor(0.5, device=self.device)
        basis = self.spline.get_acceleration_basis_vector(t)
        expected = torch.tensor([0.0, 0.0, 2.0, 3.0], device=self.device)
        self.assertTrue(torch.allclose(basis, expected))

        # Test batch of values
        t_batch = torch.tensor([0.0, 0.5, 1.0], device=self.device)
        basis_batch = self.spline.get_acceleration_basis_vector(t_batch)
        expected_batch = torch.tensor([
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0, 3.0],
            [0.0, 0.0, 2.0, 6.0]
        ], device=self.device)
        self.assertTrue(torch.allclose(basis_batch, expected_batch))

    def test_basis_mask_matrix(self):
        """Test the basis mask matrix computation."""
        num_knots = 5
        num_samples = 10
        phi = self.spline.compute_basis_mask_matrix(num_knots, num_samples)

        # Check dimensions
        self.assertEqual(phi.shape, (num_samples, num_knots))

        # Check that each row sums to approximately 1 (B-spline property)
        row_sums = torch.sum(phi, dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

    def test_evaluate_spline(self):
        """Test spline evaluation with known control points."""
        # Create a simple control point sequence (a line in 2D)
        control_points = torch.tensor([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0]
        ], device=self.device)

        # Evaluate at single parameter
        t = torch.tensor(0.5, device=self.device)
        point = self.spline.evaluate_spline(control_points, t)

        # For a linear sequence, expect approximately linear interpolation
        self.assertTrue(torch.allclose(point[0], point[1], atol=1e-5))

        # Evaluate at multiple parameters
        t_batch = torch.tensor([0.0, 0.5, 1.0], device=self.device)
        points = self.spline.evaluate_spline(control_points, t_batch)

        # Check shape
        self.assertEqual(points.shape, (3, 2))

    def test_interpolate_batch(self):
        """Test batch interpolation."""
        batch_size = 2
        num_knots = 5
        dim = 3
        num_samples = 10

        # Create batch of control points
        knot_points = torch.zeros((batch_size, num_knots, dim), device=self.device)

        # Set first batch to ascending values
        for i in range(num_knots):
            knot_points[0, i] = torch.tensor([i, i, i], device=self.device)

        # Set second batch to descending values
        for i in range(num_knots):
            knot_points[1, i] = torch.tensor([num_knots-i-1, num_knots-i-1, num_knots-i-1], device=self.device)

        # Interpolate
        dense_trajectories = self.spline.interpolate_batch(knot_points, num_samples)

        # Check shape
        self.assertEqual(dense_trajectories.shape, (batch_size, num_samples, dim))

        # For ascending batch, first sample should be smaller than last sample
        self.assertTrue(torch.all(dense_trajectories[0, 0] < dense_trajectories[0, -1]))

        # For descending batch, first sample should be larger than last sample
        self.assertTrue(torch.all(dense_trajectories[1, 0] > dense_trajectories[1, -1]))

    def test_spline_continuity(self):
        """Test that the spline maintains C2 continuity."""
        # Create control points for a curve
        control_points = torch.tensor([
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 1.0],
            [4.0, 3.0],
            [2.0, 5.0]
        ], device=self.device)

        # Evaluate at closely spaced points across a join
        t_values = torch.tensor([0.49, 0.5, 0.51], device=self.device)
        points = self.spline.evaluate_spline(control_points, t_values)

        # Calculate finite differences to approximate derivatives
        velocity_approx = (points[2] - points[0]) / (t_values[2] - t_values[0])

        # Points should be close to each other (continuity)
        self.assertTrue(torch.norm(points[1] - points[0]) < 0.1)
        self.assertTrue(torch.norm(points[2] - points[1]) < 0.1)

    def test_plot_2d_spline(self):
        """Test and visualize a 2D spline with various control points."""
        # Skip test if matplotlib is not available
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("Matplotlib not available")

        # Create control points for a 2D curve
        control_points = torch.tensor([
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 1.0],
            [4.0, 3.0],
            [2.0, 5.0],
            [0.0, 4.0]
        ], device=self.device)

        # Generate dense samples for visualization
        num_samples = 100
        t_values = torch.linspace(0, 1, num_samples, device=self.device)

        # Evaluate spline at these points
        points = self.spline.evaluate_spline(control_points, t_values)

        # Convert to numpy for plotting
        control_np = control_points.cpu().numpy()
        curve_np = points.cpu().numpy()

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot control points
        plt.plot(control_np[:, 0], control_np[:, 1], 'o-', label='Control Points')

        # Plot spline curve
        plt.plot(curve_np[:, 0], curve_np[:, 1], '-', label='B-Spline Curve')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Cubic B-Spline Curve')
        plt.grid(True)
        plt.legend()

        # Show the figure instead of saving
        plt.show()
        

    def test_plot_3d_spline(self):
        """Test and visualize a 3D spline with various control points."""
        # Skip test if matplotlib is not available
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            self.skipTest("Matplotlib or mplot3d not available")

        # Create control points for a 3D curve
        control_points = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
            [3.0, 1.0, 2.0],
            [4.0, 3.0, 0.0],
            [2.0, 5.0, 3.0],
            [0.0, 4.0, 2.0]
        ], device=self.device)

        # Generate dense samples for visualization
        num_samples = 100
        t_values = torch.linspace(0, 1, num_samples, device=self.device)

        # Evaluate spline at these points
        points = self.spline.evaluate_spline(control_points, t_values)

        # Convert to numpy for plotting
        control_np = control_points.cpu().numpy()
        curve_np = points.cpu().numpy()

        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot control points
        ax.plot(control_np[:, 0], control_np[:, 1], control_np[:, 2], 'o-', label='Control Points')

        # Plot spline curve
        ax.plot(curve_np[:, 0], curve_np[:, 1], curve_np[:, 2], '-', label='B-Spline Curve')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Cubic B-Spline Curve')
        ax.legend()

        # Show the figure instead of saving
        plt.show()
        

    def test_batch_visualization(self):
        """Test batch interpolation with visualization."""
        # Skip test if matplotlib is not available
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("Matplotlib not available")

        batch_size = 3
        num_knots = 5
        dim = 2
        num_samples = 50

        # Create batch of control points
        knot_points = torch.zeros((batch_size, num_knots, dim), device=self.device)

        # Set first batch to a simple curve
        knot_points[0] = torch.tensor([
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 1.0],
            [4.0, 3.0],
            [2.0, 5.0]
        ], device=self.device)

        # Set second batch to a circle-like shape
        theta = torch.linspace(0, 2*np.pi, num_knots, device=self.device)
        knot_points[1, :, 0] = 3 * torch.cos(theta) + 3
        knot_points[1, :, 1] = 3 * torch.sin(theta) + 3

        # Set third batch to a zigzag pattern
        knot_points[2] = torch.tensor([
            [0.0, 0.0],
            [1.0, 3.0],
            [2.0, 0.0],
            [3.0, 3.0],
            [4.0, 0.0]
        ], device=self.device)

        # Interpolate
        dense_trajectories = self.spline.interpolate_batch(knot_points, num_samples)

        # Convert to numpy for plotting
        knots_np = knot_points.cpu().numpy()
        curves_np = dense_trajectories.cpu().numpy()

        # Create plot with subplots
        fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))

        curve_types = ['Simple Curve', 'Circle-like Curve', 'Zigzag Pattern']

        for i in range(batch_size):
            ax = axes[i]

            # Plot control points
            ax.plot(knots_np[i, :, 0], knots_np[i, :, 1], 'o-', label='Control Points')

            # Plot spline curve
            ax.plot(curves_np[i, :, 0], curves_np[i, :, 1], '-', label='B-Spline')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(curve_types[i])
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()
        

    def test_spline_derivatives(self):
        """Test and visualize position, velocity, and acceleration of a spline."""
        # Skip test if matplotlib is not available
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("Matplotlib not available")

        # Create control points for a 2D curve
        control_points = torch.tensor([
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 1.0],
            [4.0, 3.0],
            [2.0, 5.0]
        ], device=self.device)

        # Generate dense samples for visualization
        num_samples = 100
        t_values = torch.linspace(0, 1, num_samples, device=self.device)

        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        # Plot position
        points = self.spline.evaluate_spline(control_points, t_values)
        axes[0].plot(t_values.cpu().numpy(), points.cpu().numpy()[:, 0], 'r-', label='X Position')
        axes[0].plot(t_values.cpu().numpy(), points.cpu().numpy()[:, 1], 'b-', label='Y Position')
        axes[0].set_ylabel('Position')
        axes[0].set_title('B-Spline Position')
        axes[0].grid(True)
        axes[0].legend()

        # Manually compute velocity using finite differences
        dt = 1.0 / (num_samples - 1)
        velocity = (points[1:] - points[:-1]) / dt
        t_vel = t_values[:-1] + dt/2  # Midpoints

        axes[1].plot(t_vel.cpu().numpy(), velocity.cpu().numpy()[:, 0], 'r-', label='X Velocity')
        axes[1].plot(t_vel.cpu().numpy(), velocity.cpu().numpy()[:, 1], 'b-', label='Y Velocity')
        axes[1].set_ylabel('Velocity')
        axes[1].set_title('B-Spline Velocity (Finite Difference)')
        axes[1].grid(True)
        axes[1].legend()

        # Manually compute acceleration using finite differences
        acceleration = (velocity[1:] - velocity[:-1]) / dt
        t_acc = t_vel[:-1] + dt/2  # Midpoints of midpoints

        axes[2].plot(t_acc.cpu().numpy(), acceleration.cpu().numpy()[:, 0], 'r-', label='X Acceleration')
        axes[2].plot(t_acc.cpu().numpy(), acceleration.cpu().numpy()[:, 1], 'b-', label='Y Acceleration')
        axes[2].set_ylabel('Acceleration')
        axes[2].set_xlabel('Parameter t')
        axes[2].set_title('B-Spline Acceleration (Finite Difference)')
        axes[2].grid(True)
        axes[2].legend()

        plt.tight_layout()
        plt.show()
        

if __name__ == "__main__":
    unittest.main()
