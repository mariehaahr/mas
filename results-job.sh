#!/bin/bash
#SBATCH --job-name=results-testrun
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=1
#SBATCH --time=00:40:00
#SBATCH --output=logs/qwen1.5/%x.%j.out
#SBATCH --error=logs/qwen1.5/%x.%j.err
#SBATCH --mail-type=END

echo "Host: $(hostname)"

set -euo pipefail

uv run src/results.py

