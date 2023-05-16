#!/bin/bash
#SBATCH --job-name=getesm
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source activate
conda activate protein_predict

# python /lustre/gst/xuchunfu/zhangxt/myesm/contact_predict.py \
#     --model mlmp00 \
#     --parameters /lustre/gst/xuchunfu/zhangxt/checkpoint/230506_token_mlp7/checkpoint-27630/pytorch_model-00002-of-00002.bin
# python /lustre/gst/xuchunfu/zhangxt/myesm/contact_predict.py \
    # --model mlmp15 \
    # --parameters /lustre/gst/xuchunfu/zhangxt/checkpoint/230330_token/checkpoint-6447/pytorch_model-00002-of-00002.bin

# python /lustre/gst/xuchunfu/zhangxt/myesm/contact_predict.py \
#     --model mlmp30 \
#     --parameters /lustre/gst/xuchunfu/zhangxt/checkpoint/230508_token_mlp3/checkpoint-12894/pytorch_model-00002-of-00002.bin

# python /lustre/gst/xuchunfu/zhangxt/myesm/contact_predict.py \
#     --model mlmp45 \
#     --parameters /lustre/gst/xuchunfu/zhangxt/checkpoint/230508_token_mlp45/checkpoint-20262/pytorch_model-00002-of-00002.bin

# python /lustre/gst/xuchunfu/zhangxt/myesm/contact_predict.py \
#     --model mlmp50 \
#     --parameters /lustre/gst/xuchunfu/zhangxt/checkpoint/230503_token_mlp5/checkpoint-9210/pytorch_model-00002-of-00002.bin

# python /lustre/gst/xuchunfu/zhangxt/myesm/contact_predict.py \
#     --model mlmp60 \
#     --parameters /lustre/gst/xuchunfu/zhangxt/checkpoint/230508_token_mlp6/checkpoint-31314/pytorch_model-00002-of-00002.bin

python /lustre/gst/xuchunfu/zhangxt/myesm/contact_predict.py \
    --model mlmp70 \
    --parameters /lustre/gst/xuchunfu/zhangxt/checkpoint/230506_token_mlp7/checkpoint-27630/pytorch_model-00002-of-00002.bin

