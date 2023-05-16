import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8"
import typing as T
from pathlib import Path
from utils import get_bb_frames
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
# from TPFold import ESMFold
# from nolgfold import ESMFold
import TPFold 
import utils
import torch.distributed as dist
import torch.optim as optim
from openfold.utils.rigid_utils import Rigid
from openfold.utils.loss import compute_fape
import noam_opt
import omegaconf
parser = argparse.ArgumentParser()
parser.add_argument('--shuffle', type=float, default=0., help='Shuffle fraction')
parser.add_argument('--data_jsonl', type=str, default="/pubhome/xtzhang/data/test_set.jsonl", help='Path for the jsonl data')
parser.add_argument('--split_json', type=str, default="/pubhome/bozhang/data/tmpnn_v8.json",help='Path for the split json file')
parser.add_argument('--output_folder',type=str,default="/pubhome/xtzhang/result/output/",help="output folder for the log files and model parameters")
parser.add_argument('--save_folder',type=str,default="/pubhome/xtzhang/result/save/",help="output folder for the model parameters")
parser.add_argument('--description',type=str,help="description the model information into wandb")
parser.add_argument('--project_name',type=str,default="TPFold",help="jobname of the wandb dashboard")
parser.add_argument('--run_name',type=str,default="eva_get_bb_frame",help="jobname of the wandb dashboard")
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
# parser.add_argument('--pdb',default="/pubhome/xtzhang/output/pdb/800aa_noseq_nocctop", help="Path to output PDB directory", type=Path, required=True)
# parser.add_argument("-o", "--pdb", help="Path to output PDB directory", type=Path, required=True)
args = parser.parse_args()
import wandb


jsonl_file = args.data_jsonl

dataset = StructureDataset(jsonl_file=jsonl_file, max_length=args.max_length) # total dataset of the pdb files


# 生成骨架
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.hub.set_dir("/pubhome/bozhang/.cache/torch/hub/")

# path_checkpoint = args.parameters
# cfg=omegaconf.dictconfig.DictConfig( 
#  content={'_name': 'ESMFoldConfig', 'esm_type': 'esm2_3B', 'fp16_esm': True, 'use_esm_attn_map': False, 'esm_ablate_pairwise': False, 'esm_ablate_sequence': False, 'esm_input_dropout': 0, 'trunk': {'_name': 'FoldingTrunkConfig', 'num_blocks': 48, 'sequence_state_dim': 1024, 'pairwise_state_dim': 128, 'sequence_head_width': 32, 'pairwise_head_width': 32, 'position_bins': 32, 'dropout': 0, 'layer_drop': 0, 'cpu_grad_checkpoint': False, 'max_recycles': 4, 'chunk_size': None, 'structure_module': {'c_s': 384, 'c_z': 128, 'c_ipa': 16, 'c_resnet': 128, 'no_heads_ipa': 12, 'no_qk_points': 4, 'no_v_points': 8, 'dropout_rate': 0.1, 'no_blocks': 8, 'no_transition_layers': 1, 'no_resnet_blocks': 2, 'no_angles': 7, 'trans_scale_factor': 10, 'epsilon': 1e-08, 'inf': 100000.0}}, 'embed_aa': True, 'bypass_lm': False, 'lddt_head_hid_dim': 128}
#  )
# model_data = torch.load(str(path_checkpoint), map_location="cpu") #读取一个pickle文件为一个dict
# from nolgfold import ESMFold
# model = ESMFold(esmfold_config=cfg) # make an instance
# model.load_state_dict(model_data['model_state_dict'], strict=False)

# # model.load_state_dict(model_data['model_state_dict'], strict=False)
# model.set_chunk_size(args.chunk_size)
# model = model.to(device)
model = TPFold.load_model()
model = model.to(device)

jsonl_file = args.data_jsonl


def initial_wandb(args):
    config = {
        "lr":args.lr,
        "epoch":args.epoch,
        "batch_size":args.batch_size,
        "chunk_size":args.chunk_size,
        "max_length":args.max_length,
        "max_tokens":args.max_length,
        "cuda_use": dist.get_world_size(),
        "device name":{i:torch.cuda.get_device_name(i) for i in range(dist.get_world_size())}
    }
    os.environ["WANDB_RUN_DIR"] = "230415"+args.run_name  # set the name of the run directory
    wandb.Settings(_service_wait=60)
    wandb.init( project=args.project_name, name=args.run_name+str(args.local_rank), config=config)


initial_wandb(args)
# wandb.watch(model, log="all", log_freq=500)

print("loading dataset")
dataset = StructureDataset(jsonl_file=jsonl_file, max_length=args.max_length) # total dataset of the pdb files
# split_file = args.split_json
# dataset_indices = {d['name']:i for i,d in enumerate(dataset)} # 每个名字对应idx
# with open(f"{split_file}","r") as f:
#     dataset_splits = json.load(f)
# train_set0, validation_set0, test_set0 = [
#         Subset(dataset, [dataset_indices[chain_name] for chain_name in dataset_splits[key]
#         if chain_name in dataset_indices]) for key in ['train', 'validation', 'test']] 
# set1=Dataset()
test_loader = DataLoader(
    dataset=dataset, 
    batch_size=args.batch_size, 
    collate_fn=batch_collate_function_nocluster)  

print("finishing dataset")
with torch.no_grad():
    for iteration, batch in enumerate(test_loader):
            
                # move train data to different gpu
                model.eval()
                for key in batch:
                    if key != 'name':
                        batch[key] = batch[key].cuda(args.local_rank)
                C_pos, CA_pos, N_pos, seq, mask, residx, bb_pos = batch['C_pos'], batch['CA_pos'], batch['N_pos'],batch['seq'], batch['mask'], batch['residx'], batch['bb_pos']
                # C_pos, CA_pos, N_pos, seq, mask, residx, bb_pos = set_grade(C_pos, CA_pos, N_pos, seq, mask, residx, bb_pos)
                
                output_dict = model(aa=seq, mask=mask, residx=residx)
                # target_frames = Rigid.from_3_points(C_pos, CA_pos, N_pos)
                target_frames = get_bb_frames(torch.stack((N_pos, CA_pos, C_pos), dim=-2))

                # target_frames = Rigid.from_3_points(C_pos, CA_pos, N_pos)
# 
                loss_fape = torch.mean(compute_fape(
                                pred_frames=output_dict['pred_frames'],
                                target_frames=target_frames,
                                frames_mask=output_dict['frame_mask'],
                                pred_positions=output_dict['backbone_positions'],
                                target_positions=bb_pos,
                                positions_mask=output_dict['backbone_atoms_mask'],
                                length_scale=10,
                            ))
                metric = {'TRAIN/loss_fape': (loss_fape).item(),
                # "TRAIN/loss": loss.item(),
                }
                wandb.log(metric)
                print(f'TRAIN: | {iteration:5d}/{len(test_loader):5d} batches | ')                # del output_dict['pred_frames']
                # output_dict['positions'] = output_dict['positions'][-1, :,:, :3, :]
                # pdbs = model.output_to_pdb(output_dict)
                # for pdb in pdbs:
                #     output_file = args.pdb / f"{iteration}.pdb"
                #     output_file.write_text(pdb)

# 补全侧链
def view_pdb(fpath, chain_ids, colors=['red','blue','green']):
        with open(fpath) as ifile:
            system = "".join([x for x in ifile])

        view = py3Dmol.view(width=600, height=400)
        view.addModelsAsFrames(system)
        for chain_id, color in zip(chain_ids, colors):
            view.setStyle({'model': -1, 'chain': chain_id}, {"cartoon": {'color': color}})
        view.zoomTo()
        view.show()

# 计算RMSD/TM-score


