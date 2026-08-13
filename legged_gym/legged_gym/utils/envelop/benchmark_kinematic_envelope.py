"""Reproducible CPU benchmark for the batched kinematic support envelope."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

import torch

from kinematic_envelope import (
    BatchedUrdfKinematics,
    capsule_support,
    default_el4090_capsules,
    load_urdf_joints,
    support_directions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--directions", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(__file__).resolve().parents[3]
    urdf = root / "resources" / "robots" / "el_4090" / "urdf" / "el_4090.urdf"
    kinematics = BatchedUrdfKinematics(load_urdf_joints(urdf))
    q = torch.linspace(-0.4, 0.4, 18, dtype=torch.float32).repeat(args.batch, 1)
    directions = support_directions(args.directions)
    capsules = default_el4090_capsules()
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with torch.inference_mode():
        for _ in range(args.warmup):
            capsule_support(kinematics, q, capsules, directions)
        timings = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            result = capsule_support(kinematics, q, capsules, directions)
            timings.append(time.perf_counter() - start)
    median = statistics.median(timings)
    report = {
        "device": "cpu",
        "dtype": str(q.dtype),
        "batch": args.batch,
        "directions": args.directions,
        "capsules": len(capsules),
        "threads": torch.get_num_threads(),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "median_ms": 1000.0 * median,
        "min_ms": 1000.0 * min(timings),
        "throughput_env_s": args.batch / median,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_rss_increase_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - baseline_rss,
        "output_shape": list(result.shape),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
