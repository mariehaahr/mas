#!/bin/bash
#SBATCH --job-name=influencing
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=1
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --mail-type=END

echo "Host: $(hostname)"

set -euo pipefail

uv run src/influencing.py

