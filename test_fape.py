import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8"
import typing as T
from pathlib import Path
import copy
from tqdm import *
import torch
from esm.esmfold.v1 import esmfold
import argparse
import json
from torch.utils.data.dataset import Subset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import time 
import wandb
from data import ClusteredDataset_inturn, StructureDataset, batch_collate_function_nocluster

import TPFold 
import utils
import torch.distributed as dist
import torch.optim as optim
from openfold.utils.rigid_utils import Rigid
from openfold.utils.loss import compute_fape,lddt_loss
import noam_opt
import omegaconf
parser = argparse.ArgumentParser()
parser.add_argument('--shuffle', type=float, default=0., help='Shuffle fraction')
parser.add_argument('--data_jsonl', type=str, default="/lustre/gst/xuchunfu/zhangxt/data/native20.jsonl", help='Path for the jsonl data')
parser.add_argument('--split_json', type=str, default="/lustre/gst/xuchunfu/zhangxt/data/.json",help='Path for the split json file')
parser.add_argument('--output_folder',type=str,default="/pubhome/xtzhang/result/output/",help="output folder for the log files and model parameters")
parser.add_argument('--save_folder',type=str,default="/pubhome/xtzhang/result/save/",help="output folder for the model parameters")
parser.add_argument('--job_name',type=str,default="noplm_eva",help="jobname of the wandb dashboard")
parser.add_argument('--num_tags',type=int,default=6,help="num tags for the sequence")
parser.add_argument('--epochs',type=int,default=5,help="epochs to train the model")
parser.add_argument('--batch_size',type=int,default=1,help="batch size tokens")
parser.add_argument('--max_length',type=int,default=800,help="max length of the training sequence")
parser.add_argument('--max_tokens',type=int,default=400,help="max length of the training sequence")
parser.add_argument('--mask',type=float,default=1.0,help="mask fractions into input sequences")
parser.add_argument("--local_rank", default=0, help="local device ID", type=int) 
parser.add_argument('--parameters',type=str,default="/pubhome/xtzhang/result/save/no_plm_or_else_384epoch4.pt", help="parameters path")
parser.add_argument('--lr',type=float,default=5e-4, help="learning rate of Adam optimizer")
parser.add_argument('--chunk_size',type=int,default=4,help="chunk size of the model")
parser.add_argument('--world_size',type=int,default=2,help="world_size")
parser.add_argument('--pattern',type=str,default="no",help="mode")
parser.add_argument('--add_tmbed',type=bool,default=False, help="whether addtmbed")
parser.add_argument('--watch_freq',type=int,default=500, help="watch gradient")
# parser.add_argument('--pdb',default="/pubhome/xtzhang/output/test_fape/", help="Path to output PDB directory", type=Path, required=True)
# parser.add_argument("-o", "--pdb", help="Path to output PDB directory", type=Path, required=True)
args = parser.parse_args()
import wandb
import nolgfold



save_path = "/pubhome/xtzhang/output/test_fape/"
# 生成骨架
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.hub.set_dir("/pubhome/bozhang/.cache/torch/hub/")

cfg=omegaconf.dictconfig.DictConfig( 
 content={'_name': 'ESMFoldConfig', 'esm_type': 'esm2_3B', 'fp16_esm': True, 'use_esm_attn_map': False, 'esm_ablate_pairwise': False, 'esm_ablate_sequence': False, 'esm_input_dropout': 0, 'trunk': {'_name': 'FoldingTrunkConfig', 'num_blocks': 48, 'sequence_state_dim': 1024, 'pairwise_state_dim': 128, 'sequence_head_width': 32, 'pairwise_head_width': 32, 'position_bins': 32, 'dropout': 0, 'layer_drop': 0, 'cpu_grad_checkpoint': False, 'max_recycles': 4, 'chunk_size': None, 'structure_module': {'c_s': 384, 'c_z': 128, 'c_ipa': 16, 'c_resnet': 128, 'no_heads_ipa': 12, 'no_qk_points': 4, 'no_v_points': 8, 'dropout_rate': 0.1, 'no_blocks': 8, 'no_transition_layers': 1, 'no_resnet_blocks': 2, 'no_angles': 7, 'trans_scale_factor': 10, 'epsilon': 1e-08, 'inf': 100000.0}}, 'embed_aa': True, 'bypass_lm': False, 'lddt_head_hid_dim': 128}
 )
print("loading lm_model")
model = TPFold.load_model()
model = model.to(device)
model.eval()
# print("loading nolm_model")
# model_no = nolgfold.load_model(chunk_size=64, pattern=args.pattern, model_path="/pubhome/xtzhang/result/save/no_plm_or_else_384epoch4.pt", cfg=cfg)
# model_no = model_no.to(device)
# model_no.eval()




print("loading dataset")
jsonl_file = args.data_jsonl
dataset = StructureDataset(jsonl_file=jsonl_file, max_length=args.max_length) 
test_loader = DataLoader(
    dataset=dataset, 
    batch_size=args.batch_size, 
    collate_fn=batch_collate_function_nocluster)  
print("finishing dataset")
from utils import get_bb_frames
ft_path="/lustre/gst/xuchunfu/zhangxt/checkpoint/230330_tokencctop_ratio2/epoch6.pt"
esm2_path="/lustre/gst/xuchunfu/zhangxt/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt"
esmfold_path="/lustre/gst/xuchunfu/zhangxt/checkpoint/TPFold/ftesm2_0425_loss4/ftesm2_0425_loss2_epoch13.pt"


print(f"esmfold_path:{esmfold_path}")
print(f"ft_path:{ft_path}")
from TPFold import load_model_old
tpmodel = load_model_old(
    esmfold_path=esmfold_path,
    esm2_path=esm2_path,
    ft_path=ft_path
)
tpmodel = tpmodel.to(device)

with torch.no_grad():
    for iteration, batch in enumerate(test_loader):            
        for key in batch:
            if key != 'name':
                batch[key] = batch[key].cuda(args.local_rank)
        C_pos, CA_pos, N_pos, seq, mask, residx, bb_pos = batch['C_pos'], batch['CA_pos'], batch['N_pos'],batch['seq'], batch['mask'], batch['residx'], batch['bb_pos']                
        # output_dict_no = model_no(aa=seq, mask=mask, residx=residx)
        target_frames_o = Rigid.from_3_points(C_pos, CA_pos, N_pos)
        target_frames_z = get_bb_frames(torch.stack((N_pos, CA_pos, C_pos), dim=-2))
        output_dict = model(aa=seq, mask=mask, residx=residx)
        tp_dict = tpmodel(aa=seq, mask=mask, residx=residx)
        loss_fape_z = torch.mean(compute_fape(
                        pred_frames=output_dict['pred_frames'],
                        target_frames=target_frames_z,
                        frames_mask=output_dict['frame_mask'],
                        pred_positions=output_dict['backbone_positions'],
                        target_positions=bb_pos,
                        positions_mask=output_dict['backbone_atoms_mask'],
                        length_scale=10,
                    ))        
        loss_plddt = lddt_loss(
                tp_logit=tp_dict['lddt_head'][-1][...,1,:],
                logits=output_dict['lddt_head'][-1][...,1,:], 
                all_atom_pred_pos=output_dict['positions'][0,:,:,:3,:], 
                all_atom_positions=torch.stack((N_pos, CA_pos, C_pos), dim=-2),
                all_atom_mask=mask.unsqueeze(-1).repeat(1,1,3), 
                plddt=output_dict['plddt'],
                tpplddt=tp_dict['plddt'],
                resolution=1.5)

                    


