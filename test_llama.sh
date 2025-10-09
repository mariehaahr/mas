#!/bin/bash
#SBATCH --job-name=environ
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/test_run/%x.%j.out
#SBATCH --constraint="gpu_a100_40gb|gpu_a100_80gb|gpu_l40s|gpu_l40s"
#SBATCH --error=logs/test_run/%x.%j.err
#SBATCH --exclude=cn8,cn14,cn15,cn16,cn17,desktop[1-16]

set -euo pipefail

echo "Host: $(hostname)"

PROJ="/home/fril/mas"
cd "$PROJ"

uv sync --python 3.12
uv remove pynvml
uv add nvidia-ml-py

# python3 -V
# python -V
# python3 /home/rp-fril-mhpe/rp/test_llama.py