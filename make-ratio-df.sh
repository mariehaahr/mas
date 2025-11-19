#!/bin/bash
#SBATCH --job-name=make-ratio2-check
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/ratio2/%x.%j.out
#SBATCH --error=logs/ratio2/%x.%j.err


echo "Host: $(hostname)"

set -euo pipefail
uv sync
uv run src/ratio.py --round 1
# uv run src/ratio.py --round 2
uv run check-ratio.py
