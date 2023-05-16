#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH -p gpu1
#SBATCH --gres=gpu:4
#SBATCH --output=/lustre/gst/xuchunfu/zhangxt/myesm/fixesm2_0516_loss2%j.out
#SBATCH --error=/lustre/gst/xuchunfu/zhangxt/myesm/fixesm2_0516_loss2%j.err

source activate
conda activate protein_predict

cd /pubhome/xtzhang/myesm/
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=54321 /lustre/gst/xuchunfu/zhangxt/myesm/fie2_loss2.py \
    --lr 5e-4 \
    --world_size 4 \
    --watch_freq 0 \
    --pattern all \
    --max_length 8000 \
    --project_name TPFold \
    --crop_size 384 \
    --batch_size 1 \
    --epoch 20 \
    --accumulation_steps 32 \
    --log_interval 128 \
    --data_jsonl /lustre/gst/xuchunfu/zhangxt/data/tmpnn_v8.jsonl \
    --split_json /lustre/gst/xuchunfu/zhangxt/data/random_v8.json \
    --ft_path /lustre/gst/xuchunfu/zhangxt/checkpoint/230508_token_mlp45/checkpoint-20262/pytorch_model-00002-of-00002.bin \
    --run_name fixesm2_0516_loss2 \
    --save_folder /lustre/gst/xuchunfu/zhangxt/checkpoint/TPFold/fixesm2_0516_loss2/ \

    

