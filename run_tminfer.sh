#!/bin/bash
#SBATCH --job-name=fold
#SBATCH -p gpu1
#SBATCH --gres=gpu:1
#SBATCH --output=/lustre/gst/xuchunfu/zhangxt/output/TPFold/ftesm2_0425_loss4_epoch13/fold%j.out
#SBATCH --error=/lustre/gst/xuchunfu/zhangxt/output/TPFold/ftesm2_0425_loss4_epoch13/fold%j.err

source activate
conda activate protein_predict

python3 /lustre/gst/xuchunfu/zhangxt/myesm/tm_inference.py \
        -i /lustre/gst/xuchunfu/zhangxt/data/test_set.fa\
        -o /lustre/gst/xuchunfu/zhangxt/output/TPFold/ftesm2_0425_loss4_epoch13/pdb/
        
python /lustre/gst/xuchunfu/zhangxt/TMalign/TMalign/TMalign.py \
        --o /lustre/gst/xuchunfu/zhangxt/output/TPFold/ftesm2_0425_loss4_epoch13/randt \
        --p /lustre/gst/xuchunfu/zhangxt/output/TPFold/ftesm2_0425_loss4_epoch13/pdb/ \
        --t /lustre/gst/xuchunfu/zhangxt/data/pdb/pdb/