#!/bin/bash
#SBATCH --job-name=third-test-2ndR
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
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
# uv run run-eval-round2.py --model_name llama-3.1-8b -limit 20 --outdir results/ # virker ikke 
# uv run run-eval-round2.py --model_name llama-3.2-1b -limit 20 --outdir results/
#uv run run-eval-round2.py --model_name qwen-2.5-7b -limit 50_000 --outdir results/
uv run run-eval-round2.py --model_name qwen-2.5-1.5b -limit 10 --outdir results/ --repetition 30
# uv run run-eval-round2.py --model_name mistral-0.3-7b -limit 10 --outdir results/
# uv run run-eval-round2.py --model_name mistral-0.2-7b -limit 10 --outdir results/

