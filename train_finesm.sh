#!/bin/bash
#SBATCH --job-name=zb_esm_mp
#SBATCH -p carbon
#SBATCH --gres=gpu:A100:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source activate
conda activate protein_design

     --output_folder parameters_ddp/ \
    --project_name "poc-esm-inpaint_v2" \
    --data_jsonl /pubhome/bozhang/data/esm_chain_set.jsonl \
    --split_json /pubhome/bozhang/data/esm_split_poc.json \
    --epochs 2 \
    --batch_size 4 \
    --max_length 300 \
    --lr 0.01 \
    --ddp_train