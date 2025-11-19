#!/bin/bash
#SBATCH --job-name=data_round2
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --mail-type=END

echo "Host: $(hostname)"

set -euo pipefail


uv run src/prepare_round2.py mistral-0.3-7b
uv run src/prepare_round2.py llama-3.2-1b
uv run src/prepare_round2.py llama-3.2-3b
#uv run src/prepare_round2.py qwen-2.5-7b
uv run src/prepare_round2.py qwen-2.5-1.5b
uv run src/prepare_round2.py mistral-0.2-7b
