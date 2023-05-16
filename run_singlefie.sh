#!/bin/bash
#SBATCH --job-name=zxt_finetuneesm
#SBATCH -p carbon
#SBATCH --gres=gpu:A100:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source activate
conda activate TMprotein_predict

python3 /pubhome/xtzhang/myesm/fie_nolg.py \
    --add_tmbed False \
    --world_size 4 \
    --watch_freq 3 \
    --pattern all \
    --max_length 400 \
    --max_tokens 400 \
    --job_name no_plm_nor_elsse\

