#!/bin/bash
#SBATCH --job-name=first-test
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --constraint="gpu_rtx8000|gpu_rtx6000|gpu_l40s|gpu_a100_40gb"
#SBATCH --mail-type=END

echo "Host: $(hostname)"

set -euo pipefail


# export HF_HOME=$HOME/.cache/huggingface #run offline
# export HF_HUB_OFFLINE=1 # run offline
# export TRANSFORMERS_OFFLINE=1

uv sync 
# uv run run-eval.py --model_name llama-3.1-8b -limit 20 --outdir results/ # virker ikke 
# uv run run-eval.py --model_name llama-3.2-1b -limit 20 --outdir results/
uv run run-eval.py --model_name qwen-2.5-7b -limit 50_000 --outdir results/
# uv run run-eval.py --model_name qwen-2.5-1.5b -limit 20 --outdir results/
# uv run run-eval.py --model_name mistral-0.3-7b -limit 50_000 --outdir results/
# uv run src/structured-output.py
