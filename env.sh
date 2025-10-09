#!/bin/bash
#SBATCH --job-name=uv-env
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --exclude=cn8,cn14,cn15,cn16,cn17,desktop[1-16]


echo "Host: $(hostname)"

PROJ="/home/rp-fril-mhpe/rp"

cd "$PROJ"


uv add bitsandbytes==0.48.1
uv add flashinfer-python==0.3.1.post1

uv sync 
uv lock 

