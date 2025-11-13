#!/bin/bash
#SBATCH --job-name=first-mistral-0.2-7b-2
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=25:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --constraint="gpu_h100|gpu_a100_80gb"
#SBATCH --mail-type=BEGIN,END

echo "Host: $(hostname)"

set -euo pipefail

uv sync 
# uv run run-eval.py --model_name llama-3.1-8b -limit 20 --outdir results/ --repetition 1 #virker ikke 

# uv run run-eval.py --model_name llama-3.2-1b --outdir results_extra/ --repetition 10 --dataset_path data/sarc/sarcasm2_llama.csv

# uv run run-eval.py --model_name llama-3.2-3b --outdir results/ --repetition 10 --dataset_path data/sarc/sarcasm2.csv

# uv run run-eval.py --model_name qwen-2.5-7b --repetition 10 --outdir results/ --dataset_path data/sarc/sarcasm50k.csv
# uv run run-eval.py --model_name qwen-2.5-7b --repetition 10 --outdir results_extra/ --dataset_path data/sarc/sarcasm2-minus-50k.csv

# uv run run-eval.py --model_name qwen-2.5-1.5b --repetition 10 --outdir results_extra/ --dataset_path data/sarc/sarcasm2-minus-50k.csv

# uv run run-eval.py --model_name mistral-0.3-7b --outdir results/ --repetition 10

uv run run-eval.py --model_name mistral-0.2-7b --outdir results/ --repetition 10 --dataset_path data/sarc/sarcasm2-mistral2.csv

