#!/bin/bash
#SBATCH --job-name=vllm-structured
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --constraint="gpu_rtx8000|gpu_rtx6000|gpu_l40s|gpu_a100_40gb"

set -euo pipefail

uv sync 
uv run src/simple-job.py
# uv run src/structured-output.py