"""
GPU inference speed benchmark for S0-like neural network policies.
Tests maximum inference throughput for full-body control networks.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Tuple, Dict
import json
from datetime import datetime


class S0PolicyNetwork(nn.Module):
    """
    Simplified S0-like policy network for full-body control.
    - Input: full-body joint state + base motion (~100 dims)
    - Output: joint-level actuator commands (~50 dims)
    - 10M parameters (approximate, can be scaled)
    """
    
    def __init__(self, input_dim: int = 100, output_dim: int = 50, hidden_dim: int = 1024):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Main policy network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class InferenceSpeedTester:
    """Comprehensive GPU inference speed benchmarking tool."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.results = {}
        
    def warmup(self, model: nn.Module, input_tensor: torch.Tensor, num_iterations: int = 10):
        """Warmup GPU by running inference multiple times."""
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_tensor)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
    
    def benchmark_latency(self, 
                         model: nn.Module, 
                         input_shape: Tuple[int, ...],
                         num_iterations: int = 1000,
                         batch_size: int = 1) -> Dict[str, float]:
        """
        Measure per-inference latency (time to run one forward pass).
        """
        # Create input tensor
        input_tensor = torch.randn(*input_shape, device=self.device)
        
        # Warmup
        self.warmup(model, input_tensor)
        
        # Benchmark
        model.eval()
        with torch.no_grad():
            # Synchronize before timing
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            for _ in range(num_iterations):
                _ = model(input_tensor)
            
            # Synchronize after timing
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
        
        total_time = end_time - start_time
        latency_ms = (total_time / num_iterations) * 1000
        throughput_hz = (num_iterations / total_time)
        
        return {
            "latency_ms": latency_ms,
            "throughput_hz": throughput_hz,
            "throughput_khz": throughput_hz / 1000,
            "total_time_s": total_time,
            "num_iterations": num_iterations,
            "batch_size": batch_size,
        }
    
    def benchmark_batch_throughput(self,
                                   model: nn.Module,
                                   input_dim: int,
                                   output_dim: int,
                                   batch_sizes: list = None,
                                   num_iterations: int = 100) -> Dict[int, Dict]:
        """
        Measure throughput across different batch sizes.
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32, 64, 128]
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"  Benchmarking batch_size={batch_size}...", end=" ", flush=True)
            input_tensor = torch.randn(batch_size, input_dim, device=self.device)
            
            benchmark_result = self.benchmark_latency(
                model, 
                input_tensor.shape,
                num_iterations=num_iterations,
                batch_size=batch_size
            )
            results[batch_size] = benchmark_result
            
            print(f"✓ {benchmark_result['throughput_hz']:.0f} inferences/sec")
        
        return results
    
    def benchmark_different_architectures(self,
                                         input_dim: int = 100,
                                         output_dim: int = 50,
                                         num_iterations: int = 500):
        """
        Benchmark different network sizes to understand scaling.
        """
        hidden_dims = [256, 512, 1024, 2048, 4096]
        results = {}
        
        print(f"\nBenchmarking different network sizes:")
        print(f"{'Hidden Dim':<15} {'Parameters':<15} {'Latency (ms)':<15} {'Throughput (Hz)':<20}")
        print("-" * 65)
        
        for hidden_dim in hidden_dims:
            model = S0PolicyNetwork(input_dim, output_dim, hidden_dim).to(self.device)
            num_params = model.count_parameters()
            
            benchmark_result = self.benchmark_latency(
                model,
                (1, input_dim),
                num_iterations=num_iterations
            )
            
            results[hidden_dim] = {
                "parameters": num_params,
                **benchmark_result
            }
            
            print(f"{hidden_dim:<15} {num_params:<15,} {benchmark_result['latency_ms']:<15.4f} {benchmark_result['throughput_hz']:<20.0f}")
        
        return results
    
    def benchmark_real_time_feasibility(self, 
                                       model: nn.Module,
                                       target_frequency_hz: float = 1000.0):
        """
        Check if inference can meet real-time requirements (e.g., 1 kHz for S0).
        """
        input_dim = model.input_dim
        
        benchmark_result = self.benchmark_latency(
            model,
            (1, input_dim),
            num_iterations=1000
        )
        
        latency_ms = benchmark_result["latency_ms"]
        budget_ms = 1000.0 / target_frequency_hz
        feasible = latency_ms < budget_ms
        margin_percent = ((budget_ms - latency_ms) / budget_ms) * 100
        
        return {
            "target_frequency_hz": target_frequency_hz,
            "budget_ms": budget_ms,
            "actual_latency_ms": latency_ms,
            "feasible": feasible,
            "margin_percent": margin_percent if feasible else -margin_percent,
            **benchmark_result
        }


def main():
    """Run comprehensive inference speed tests."""
    
    print("=" * 80)
    print("GPU Inference Speed Benchmark for S0-like Neural Network Policies")
    print("=" * 80)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Capability: {torch.cuda.get_device_capability(0)}")
        device = "cuda"
    else:
        print(f"\n⚠ CUDA not available, using CPU")
        device = "cpu"
    
    tester = InferenceSpeedTester(device=device)
    
    # Create S0-like model (10M parameters)
    print(f"\nCreating S0-like policy network (10M parameters)...")
    model = S0PolicyNetwork(input_dim=100, output_dim=50, hidden_dim=2048)
    num_params = model.count_parameters()
    print(f"  Total parameters: {num_params:,}")
    
    model = model.to(device)
    model.eval()
    
    # Test 1: Single inference latency
    print(f"\n{'='*80}")
    print("Test 1: Single Inference Latency (1 kHz control loop requirement)")
    print(f"{'='*80}")
    
    realtime_result = tester.benchmark_real_time_feasibility(model, target_frequency_hz=1000.0)
    
    print(f"\nTarget frequency: {realtime_result['target_frequency_hz']} Hz (S0 control loop)")
    print(f"Time budget: {realtime_result['budget_ms']:.4f} ms")
    print(f"Actual latency: {realtime_result['actual_latency_ms']:.4f} ms")
    print(f"Feasible: {'✓ YES' if realtime_result['feasible'] else '✗ NO'}")
    if realtime_result['feasible']:
        print(f"Margin: {realtime_result['margin_percent']:.2f}% slack")
    else:
        print(f"Shortfall: {abs(realtime_result['margin_percent']):.2f}% over budget")
    print(f"Actual throughput: {realtime_result['throughput_hz']:.0f} inferences/sec")
    
    # Test 2: Batch processing throughput
    print(f"\n{'='*80}")
    print("Test 2: Batch Processing Throughput (Simulation data collection)")
    print(f"{'='*80}")
    print(f"\nWith 200,000 parallel environments, batch processing is critical:")
    
    batch_results = tester.benchmark_batch_throughput(
        model,
        input_dim=100,
        output_dim=50,
        batch_sizes=[1, 4, 8, 16, 32, 64, 128, 256],
        num_iterations=200
    )
    
    # Show best throughput
    best_batch = max(batch_results.items(), key=lambda x: x[1]["throughput_hz"])
    print(f"\n  Best throughput: {best_batch[1]['throughput_hz']:.0f} inferences/sec")
    print(f"  Achieved at batch_size={best_batch[0]}")
    
    # Test 3: Architecture scaling
    print(f"\n{'='*80}")
    print("Test 3: Architecture Scaling (Network size vs. inference speed)")
    print(f"{'='*80}")
    
    arch_results = tester.benchmark_different_architectures(
        input_dim=100,
        output_dim=50,
        num_iterations=500
    )
    
    # Test 4: Device memory usage
    print(f"\n{'='*80}")
    print("Test 4: Memory Usage")
    print(f"{'='*80}")
    
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Forward pass
        dummy_input = torch.randn(1, 100, device=device)
        _ = model(dummy_input)
        
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        
        print(f"\nModel parameters: {num_params:,}")
        print(f"Estimated model size: {num_params * 4 / (1024**2):.2f} MB (float32)")
        print(f"Peak memory (single inference): {peak_memory / (1024**2):.2f} MB")
        
        # Batch memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        batch_input = torch.randn(128, 100, device=device)
        _ = model(batch_input)
        torch.cuda.synchronize()
        peak_memory_batch = torch.cuda.max_memory_allocated()
        
        print(f"Peak memory (batch=128): {peak_memory_batch / (1024**2):.2f} MB")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print(f"\n1. Real-time Performance (1 kHz):")
    if realtime_result['feasible']:
        print(f"   ✓ Network CAN meet 1 kHz control requirements")
        print(f"   ✓ Per-inference latency: {realtime_result['actual_latency_ms']:.4f} ms")
        print(f"   ✓ Suitable for S0 deployment on real robots")
    else:
        print(f"   ✗ Network CANNOT meet 1 kHz requirements")
        print(f"   ✗ Would need to reduce network size or optimize")
    
    print(f"\n2. Simulation Training (200K parallel envs):")
    throughput_hz = best_batch[1]["throughput_hz"]
    print(f"   ✓ Max batch throughput: {throughput_hz:.0f} inferences/sec")
    print(f"   ✓ Can process {throughput_hz / 1000:.1f}K environments/sec at 1 kHz")
    
    print(f"\n3. Deployment Recommendation:")
    if realtime_result['feasible']:
        print(f"   ✓ Ready for deployment on:")
        print(f"     - NVIDIA Jetson devices (with margin)")
        print(f"     - Standard GPU servers")
        print(f"     - Consider using batch inference for offline analysis")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
