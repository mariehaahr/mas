#!/bin/bash
#SBATCH --job-name=first-qwen-2.5-7b
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --constraint="gpu_h100|gpu_a100_80gb"
#SBATCH --mail-type=BEGIN,END

echo "Host: $(hostname)"

set -euo pipefail

uv sync 
# uv run run-eval.py --model_name llama-3.1-8b -limit 20 --outdir results/ --repetition 1 #virker ikke 

# uv run run-eval.py --model_name llama-3.2-1b --outdir results/ --repetition 20 --batch_size 2048 # have already run this for 20h 
# uv run run-eval.py --model_name llama-3.2-3b --outdir results_small/ --repetition 30 -limit 10

uv run run-eval.py --model_name qwen-2.5-7b --repetition 10 --outdir results/ --dataset_path data/sarc/sarcasm2.csv

# uv run run-eval.py --model_name qwen-2.5-1.5b -limit 20 --outdir results/
# uv run run-eval.py --model_name mistral-0.3-7b -limit 1 --outdir results/ --repetition 10
# uv run src/structured-output.py
