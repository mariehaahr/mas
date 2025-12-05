#!/bin/bash
#SBATCH --job-name=plots-analysis
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

echo "Host: $(hostname)"

set -euo pipefail

uv run src/analysis.py
# uv run src/concat_results.py

#uv run src/dd.py
