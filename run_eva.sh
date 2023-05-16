#!/bin/bash
#SBATCH --job-name=finetuneesm
#SBATCH -p carbon
#SBATCH --gres=gpu:A100:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source activate
conda activate TMprotein_predict

python /pubhome/xtzhang/myesm/eva.py