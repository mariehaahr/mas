#!/bin/bash
#SBATCH --job-name=test_offline_llama_8B
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --constraint="gpu_a100_40gb|gpu_a100_80gb|gpu_l40s|gpu_a30"
#SBATCH --mail-type=END

echo "Host: $(hostname)"

set -euo pipefail

source /home/mhpe/.env

export HF_HOME=/home/mhpe/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


#uv sync 
uv run models/llama/llama_8B.py
# uv run src/structured-output.py



