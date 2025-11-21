#!/bin/bash
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#SBATCH --job-name=qwen-2.5-1.5b-2ndrun
=======
#SBATCH --job-name=llama-3.2-7b-2ndrun
>>>>>>> 41e1257 (update second-run.sh to llama)
=======
#SBATCH --job-name=mistral-0.2-7b-2ndrun
>>>>>>> 478a8a9 (ready to run mistral-0.2-7b for round 2)
=======
#SBATCH --job-name=mistral-0.3-7b-2ndrun
>>>>>>> 552cd01 (ready to run mistral-0.3 ready for round 2 again, after fixing bug)
=======
#SBATCH --job-name=qwen-2.5-1.5b-2ndrun
>>>>>>> 4c97198 (nn)
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=18:00:00
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
# already done: mistral-0.3-7b and qwen-2.5-7b
uv run run-eval-round2.py --model_name llama-3.2-3b --outdir /home/rp-fril-mhpe --dataset_path /home/rp-fril-mhpe/input_llama-3.2-3b.csv
>>>>>>> f178642 (ready for round 2 rest, llama-3.2-3b)


uv run run-eval-round2.py --model_name qwen-2.5-1.5b --outdir /home/rp-fril-mhpe --dataset_path /home/rp-fril-mhpe/input_qwen-2.5-1.5b.csv
=======
#uv run run-eval-round2.py --model_name qwen-2.5-1.5b --outdir /home/rp-fril-mhpe --dataset_path /home/rp-fril-mhpe/input_qwen-2.5-1.5b.csv
<<<<<<< HEAD
uv run run-eval-round2.py --model_name mistral-0.2-7b --outdir /home/rp-fril-mhpe --dataset_path /home/rp-fril-mhpe/input_mistral-0.2-7b.csv
>>>>>>> 478a8a9 (ready to run mistral-0.2-7b for round 2)
=======
uv run run-eval-round2.py --model_name mistral-0.3-7b --outdir /home/rp-fril-mhpe --dataset_path /home/rp-fril-mhpe/input_mistral-0.3-7b.csv
>>>>>>> 552cd01 (ready to run mistral-0.3 ready for round 2 again, after fixing bug)
=======
uv run run-eval-round2.py --model_name qwen-2.5-1.5b --outdir /home/rp-fril-mhpe --dataset_path /home/rp-fril-mhpe/input_qwen-2.5-1.5b.csv
#uv run run-eval-round2.py --model_name mistral-0.3-7b --outdir /home/rp-fril-mhpe --dataset_path /home/rp-fril-mhpe/input_mistral-0.3-7b.csv
>>>>>>> 4c97198 (nn)
