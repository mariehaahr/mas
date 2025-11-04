#!/bin/bash
#SBATCH --job-name=test-llama
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/test_run/%x.%j.out
#SBATCH --constraint="gpu_h100|gpu_a100_80gb"
#SBATCH --error=logs/test_run/%x.%j.err


set -euo pipefail

echo "Host: $(hostname)"

PROJ="/home/fril/mas"
cd "$PROJ"

uv run test_llama.py

# python3 -V
# python -V
# python3 /home/rp-fril-mhpe/rp/test_llama.py