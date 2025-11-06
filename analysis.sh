#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

echo "Host: $(hostname)"

set -euo pipefail

uv run src/analysis.py
