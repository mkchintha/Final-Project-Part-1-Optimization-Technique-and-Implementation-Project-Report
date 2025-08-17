"""
Optimization Demo: AoS vs SoA with NumPy and Numba

Single-file benchmark to demonstrate how data layout and vectorization affect
performance in an HPC-flavored particle update workload.

What this file includes
1) Baseline AoS (Array of Structures) using a NumPy structured array and a pure
   Python loop. This is intentionally slow and cache unfriendly.
2) Optimized SoA (Structure of Arrays) with two approaches:
   a) Vectorized NumPy operations
   b) Numba-compiled kernels (@njit) for fast loops that exploit contiguous data
3) A benchmark harness with warmup, multiple trials, scaling over N, and charts.
4) Deterministic random generation via a fixed seed unless overridden.
5) Outputs: a CSV of results and one or more PNG charts.

How to run
$ python optimization_demo.py --sizes 50000 100000 200000 --steps 50 --trials 5 \
    --outdir results

Dependencies
- Python 3.9+
- numpy
- numba
- matplotlib
- pandas (for CSV output)

Notes
- If you see very slow Baseline timing at large N, that is expected.
- The Numba path will trigger a JIT compile on first call; warmup is done to avoid
  polluting the measured runs.
- Keep an eye on memory when pushing N to very large values.
"""

from __future__ import annotations
import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange

# -----------------------------
# Data generation and utilities
# -----------------------------

@dataclass
class Config:
    sizes: List[int]
    steps: int
    trials: int
    dt: float
    seed: int
    outdir: Path
    no_plots: bool
    use_parallel: bool


def set_seed(seed: int) -> None:
    np.random.seed(seed)


# --------------
# AoS Baseline
# --------------

def make_aos(N: int) -> np.ndarray:
    """Create an Array-of-Structures layout using a NumPy structured dtype.

    Each particle has fields: x, y, z, vx, vy, vz. Positions and velocities.
    """
    dtype = np.dtype([
        ("x", np.float64), ("y", np.float64), ("z", np.float64),
        ("vx", np.float64), ("vy", np.float64), ("vz", np.float64),
    ])
    arr = np.zeros(N, dtype=dtype)
    arr["x"] = np.random.randn(N)
    arr["y"] = np.random.randn(N)
    arr["z"] = np.random.randn(N)
    arr["vx"] = np.random.randn(N) * 0.01
    arr["vy"] = np.random.randn(N) * 0.01
    arr["vz"] = np.random.randn(N) * 0.01
    return arr


def baseline_aos_step(arr: np.ndarray, dt: float) -> Tuple[float, float]:
    """Pure Python loop update for AoS.

    Updates positions by velocities, then computes kinetic energy sum and
    average distance from origin. Returns (kinetic_energy, mean_distance).
    """
    N = arr.shape[0]
    ke_sum = 0.0
    dist_sum = 0.0
    for i in range(N):
        arr[i]["x"] += arr[i]["vx"] * dt
        arr[i]["y"] += arr[i]["vy"] * dt
        arr[i]["z"] += arr[i]["vz"] * dt
        vx = arr[i]["vx"]; vy = arr[i]["vy"]; vz = arr[i]["vz"]
        ke_sum += 0.5 * (vx*vx + vy*vy + vz*vz)
        x = arr[i]["x"]; y = arr[i]["y"]; z = arr[i]["z"]
        dist_sum += math.sqrt(x*x + y*y + z*z)
    return ke_sum, dist_sum / N


def baseline_aos_run(N: int, steps: int, dt: float) -> Dict[str, float]:
    arr = make_aos(N)
    ke = 0.0
    md = 0.0
    for _ in range(steps):
        ke, md = baseline_aos_step(arr, dt)
    return {"ke": ke, "mean_dist": md}


# --------------
# SoA Optimized
# --------------

def make_soa(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a Structure-of-Arrays layout as 6 separate contiguous arrays."""
    x = np.random.randn(N).astype(np.float64)
    y = np.random.randn(N).astype(np.float64)
    z = np.random.randn(N).astype(np.float64)
    vx = (np.random.randn(N) * 0.01).astype(np.float64)
    vy = (np.random.randn(N) * 0.01).astype(np.float64)
    vz = (np.random.randn(N) * 0.01).astype(np.float64)
    return x, y, z, vx, vy, vz


# Vectorized NumPy version

def numpy_soa_step(x, y, z, vx, vy, vz, dt: float) -> Tuple[float, float]:
    x += vx * dt
    y += vy * dt
    z += vz * dt
    ke = 0.5 * (vx*vx + vy*vy + vz*vz)
    ke_sum = float(np.sum(ke))
    dist = np.sqrt(x*x + y*y + z*z)
    mean_dist = float(np.mean(dist))
    return ke_sum, mean_dist


def numpy_soa_run(N: int, steps: int, dt: float) -> Dict[str, float]:
    x, y, z, vx, vy, vz = make_soa(N)
    ke = 0.0
    md = 0.0
    for _ in range(steps):
        ke, md = numpy_soa_step(x, y, z, vx, vy, vz, dt)
    return {"ke": ke, "mean_dist": md}


# Numba JIT kernels

@njit(fastmath=True)
def _numba_soa_step(x, y, z, vx, vy, vz, dt):
    N = x.shape[0]
    ke_sum = 0.0
    dist_sum = 0.0
    for i in range(N):
        x[i] += vx[i] * dt
        y[i] += vy[i] * dt
        z[i] += vz[i] * dt
        vi2 = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]
        ke_sum += 0.5 * vi2
        xi = x[i]; yi = y[i]; zi = z[i]
        dist_sum += math.sqrt(xi*xi + yi*yi + zi*zi)
    return ke_sum, dist_sum / N


@njit(fastmath=True, parallel=True)
def _numba_soa_step_parallel(x, y, z, vx, vy, vz, dt):
    N = x.shape[0]
    ke_sum = 0.0
    dist_sum = 0.0
    # Parallel reduction; simple form for demonstration. In practice, use
    # local accumulators to reduce contention, then combine.
    ke_local = 0.0
    dist_local = 0.0
    for i in prange(N):
        x[i] += vx[i] * dt
        y[i] += vy[i] * dt
        z[i] += vz[i] * dt
        vi2 = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]
        ke_local += 0.5 * vi2
        xi = x[i]; yi = y[i]; zi = z[i]
        dist_local += math.sqrt(xi*xi + yi*yi + zi*zi)
    ke_sum += ke_local
    dist_sum += dist_local
    return ke_sum, dist_sum / N


def numba_soa_run(N: int, steps: int, dt: float, use_parallel: bool = False) -> Dict[str, float]:
    x, y, z, vx, vy, vz = make_soa(N)
    ke = 0.0
    md = 0.0
    step_fn = _numba_soa_step_parallel if use_parallel else _numba_soa_step
    # Warmup to trigger JIT
    ke, md = step_fn(x, y, z, vx, vy, vz, dt)
    for _ in range(steps - 1):
        ke, md = step_fn(x, y, z, vx, vy, vz, dt)
    return {"ke": ke, "mean_dist": md}


# -----------------
# Benchmark harness
# -----------------

@dataclass
class BenchResult:
    impl: str
    N: int
    steps: int
    mean_ms: float
    std_ms: float
    last_ke: float
    last_mean_dist: float


def timeit_ms(fn, trials: int) -> Tuple[float, float]:
    """Return mean and std dev in ms over given number of trials."""
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def run_one_size(N: int, steps: int, dt: float, trials: int, use_parallel: bool) -> List[BenchResult]:
    results: List[BenchResult] = []

    # Baseline AoS
    def run_baseline():
        return baseline_aos_run(N, steps, dt)
    mean_ms, std_ms = timeit_ms(run_baseline, trials)
    out = run_baseline()
    results.append(BenchResult("baseline_aos_python", N, steps, mean_ms, std_ms, out["ke"], out["mean_dist"]))

    # NumPy SoA
    def run_numpy():
        return numpy_soa_run(N, steps, dt)
    mean_ms, std_ms = timeit_ms(run_numpy, trials)
    out = run_numpy()
    results.append(BenchResult("optimized_soa_numpy", N, steps, mean_ms, std_ms, out["ke"], out["mean_dist"]))

    # Numba SoA
    def run_numba():
        return numba_soa_run(N, steps, dt, use_parallel=use_parallel)
    # Ensure JIT warmup is not counted: numba_soa_run already warms within
    # but to be safe, call once before timing
    _ = run_numba()
    mean_ms, std_ms = timeit_ms(run_numba, trials)
    out = run_numba()
    tag = "optimized_soa_numba_parallel" if use_parallel else "optimized_soa_numba"
    results.append(BenchResult(tag, N, steps, mean_ms, std_ms, out["ke"], out["mean_dist"]))

    return results


def plot_speed(results_df: pd.DataFrame, outdir: Path) -> None:
    pivot = results_df.pivot(index="N", columns="impl", values="mean_ms")
    plt.figure()
    pivot.plot(marker="o")
    plt.xlabel("N (particles)")
    plt.ylabel("Mean time per run (ms)")
    plt.title("AoS vs SoA Performance")
    plt.grid(True)
    plt.tight_layout()
    out = outdir / "speed_vs_N.png"
    plt.savefig(out, dpi=140)
    plt.close()


def plot_speedup(results_df: pd.DataFrame, outdir: Path) -> None:
    # Compute speedup relative to baseline for each N and impl
    base = results_df[results_df["impl"] == "baseline_aos_python"][["N", "mean_ms"]]
    base = base.rename(columns={"mean_ms": "baseline_ms"})
    merged = results_df.merge(base, on="N")
    merged["speedup"] = merged["baseline_ms"] / merged["mean_ms"]
    merged = merged[merged["impl"] != "baseline_aos_python"]

    pivot = merged.pivot(index="N", columns="impl", values="speedup")
    plt.figure()
    pivot.plot(marker="o")
    plt.xlabel("N (particles)")
    plt.ylabel("Speedup vs baseline")
    plt.title("Speedup of SoA approaches over AoS baseline")
    plt.grid(True)
    plt.tight_layout()
    out = outdir / "speedup_vs_N.png"
    plt.savefig(out, dpi=140)
    plt.close()


# -----------
# Main entry
# -----------

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="AoS vs SoA optimization demo")
    p.add_argument("--sizes", nargs="+", type=int, default=[20000, 50000, 100000],
                   help="Problem sizes N to benchmark")
    p.add_argument("--steps", type=int, default=50, help="Time steps per run")
    p.add_argument("--trials", type=int, default=5, help="Trials per implementation")
    p.add_argument("--dt", type=float, default=1.0, help="Timestep delta")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--outdir", type=Path, default=Path("results"), help="Output directory")
    p.add_argument("--no-plots", action="store_true", help="Disable chart generation")
    p.add_argument("--parallel", action="store_true", help="Use Numba parallel kernel")
    args = p.parse_args()
    return Config(
        sizes=args.sizes,
        steps=args.steps,
        trials=args.trials,
        dt=args.dt,
        seed=args.seed,
        outdir=args.outdir,
        no_plots=args.no_plots,
        use_parallel=args.parallel,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    print("Running benchmarks...\n")
    print(f"sizes={cfg.sizes} steps={cfg.steps} trials={cfg.trials} dt={cfg.dt} parallel={cfg.use_parallel}")

    for N in cfg.sizes:
        print(f"\nN={N}")
        res = run_one_size(N, cfg.steps, cfg.dt, cfg.trials, cfg.use_parallel)
        for r in res:
            print(f"  impl={r.impl:26s} mean={r.mean_ms:10.2f} ms  std={r.std_ms:7.2f} ms  ke={r.last_ke:.3e}  md={r.last_mean_dist:.3f}")
            rows.append({
                "impl": r.impl,
                "N": r.N,
                "steps": r.steps,
                "mean_ms": r.mean_ms,
                "std_ms": r.std_ms,
                "last_ke": r.last_ke,
                "last_mean_dist": r.last_mean_dist,
            })

    df = pd.DataFrame(rows)
    csv_path = cfg.outdir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    if not cfg.no_plots:
        plot_speed(df, cfg.outdir)
        plot_speedup(df, cfg.outdir)
        print(f"Saved plots to {cfg.outdir}")

    # Save a small README with run configuration
    with open(cfg.outdir / "README.txt", "w", encoding="utf-8") as f:
        f.write("AoS vs SoA Optimization Demo\n")
        f.write(f"sizes={cfg.sizes}\nsteps={cfg.steps}\ntrials={cfg.trials}\ndt={cfg.dt}\nseed={cfg.seed}\n")
        f.write(f"parallel={cfg.use_parallel}\n")
        f.write("\nFiles produced:\n")
        f.write("- results.csv\n")
        if not cfg.no_plots:
            f.write("- speed_vs_N.png\n- speedup_vs_N.png\n")


if __name__ == "__main__":
    main()
