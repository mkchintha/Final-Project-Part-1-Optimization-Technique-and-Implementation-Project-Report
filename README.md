# Optimization in High-Performance Computing: AoS vs SoA

## Overview
This repository contains the implementation, benchmarks, and results for the project *Optimization in High-Performance Computing: AoS vs SoA*. The project demonstrates how different data layouts (Array of Structures, AoS, vs Structure of Arrays, SoA) impact performance in high-performance computing workloads.

The implementation uses Python, NumPy, and Numba to compare:
- Baseline AoS (Python loop with structured arrays)
- Optimized SoA with NumPy (vectorized operations)
- Optimized SoA with Numba JIT (compiled loops with fast math)

Benchmarks validate the importance of cache locality and vectorization in HPC contexts.

## File Structure
- `optimization.py` — main Python script containing all implementations and the benchmark harness
- `results/`
  - `results.csv` — benchmark data (mean runtime, std dev, energy, distance)
  - `speed_vs_N.png` — absolute runtime comparison
  - `speedup_vs_N.png` — relative speedup over baseline
- `report/` — APA 7 formatted project report
- `appendix/` — Word document with code, screenshots, and GitHub repo link

## Requirements
- Python 3.9+
- NumPy
- Numba
- Matplotlib
- Pandas

Install dependencies:
```bash
pip install numpy numba matplotlib pandas
```

## Getting Started
Clone the repository and run a quick test:
```bash
git clone <your-repo-url>.git
cd <repo-folder>
python optimization_demo.py --sizes 50000 100000 200000 --steps 50 --trials 5 --outdir results
```

Optional flags:
- `--parallel` to enable Numba parallel kernel
- `--no-plots` to skip plot generation

Outputs include `results.csv` and performance plots saved under the output directory.

## Results Summary
- AoS baseline: seconds per run at N >= 100k
- NumPy SoA: about 400x to 800x faster due to vectorization
- Numba SoA: about 1800x faster due to JIT compilation
- Results confirm improved cache locality and vectorization efficiency with SoA

## Reproducibility
- Deterministic runs via fixed random seed (configurable)
- Warm-up performed before timed trials to avoid JIT skew
- Multiple trials with mean and standard deviation reported

## Acknowledgments
Developed by Murali Krishna Chintha for Algorithms and Data Structures (MSCS-532-B01) Second Bi-term.

## License
Add a license here if required (for example, MIT License).
