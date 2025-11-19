#!/bin/bash
#SBATCH --job-name=mistral-0.3-7b-2ndrun
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --constraint="gpu_h100|gpu_a100_80gb"
#SBATCH --mail-type=END

echo "Host: $(hostname)"

set -euo pipefail


# export HF_HOME=$HOME/.cache/huggingface #run offline
# export HF_HUB_OFFLINE=1 # run offline
# export TRANSFORMERS_OFFLINE=1

uv sync 

#uv run run-eval-round2.py --model_name mistral-0.3-7b --outdir /home/rp-fril-mhpe --dataset_path /home/rp-fril-mhpe/input_mistral-0.3-7b.csv
# already done: mistral-0.3-7b and qwen-2.5-7b
uv run run-eval-round2.py --model_name llama-3.2-3b --outdir /home/rp-fril-mhpe --dataset_path /home/rp-fril-mhpe/input_llama-3.2-3b.csv

