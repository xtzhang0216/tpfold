import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8"
# os.environ['CUDA_VISIVLE_DEVICES']=0,1 #这一行注意去掉！！！
import sys
import typing as T

from tqdm import *
import torch

import argparse
import json
from torch.utils.data.dataset import Subset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import time 
import wandb

from data import StructureDataset, ClusteredDataset, batch_collate_function
import TPFold 
import utils
import torch.distributed as dist
import torch.optim as optim
from openfold.utils.rigid_utils import Rigid
from openfold.utils.loss import compute_fape
import noam_opt
parser = argparse.ArgumentParser()

def get_args():
    parser.add_argument('--shuffle', type=float, default=0., help='Shuffle fraction')
    parser.add_argument('--data_jsonl', type=str, default="/pubhome/xtzhang/data/content2.jsonl", help='Path for the jsonl data')
    parser.add_argument('--split_json', type=str, default="/pubhome/xtzhang/data/split2.json",help='Path for the split json file')
    parser.add_argument('--output_folder',type=str,default="/data/home/scv6707/run/zxt/TPFold/output/",help="output folder for the log files and model parameters")
    parser.add_argument('--description',type=str,help="description the model information into wandb")
    parser.add_argument('--job_name',type=str,default="TPFold_try1",help="jobname of the wandb dashboard")
    parser.add_argument('--num_tags',type=int,default=5,help="num tags for the sequence")
    parser.add_argument('--epochs',type=int,default=1,help="epochs to train the model")
    parser.add_argument('--batch_size',type=int,default=1,help="batch size tokens")
    parser.add_argument('--max_length',type=int,default=1300,help="max length of the training sequence")
    parser.add_argument('--mask',type=float,default=1.0,help="mask fractions into input sequences")
    parser.add_argument("--local_rank", help="local device ID", type=int) 
    parser.add_argument('--parameters',type=str,default="/root/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt", help="parameters path")
    parser.add_argument('--lr',type=float,default=1e-3, help="learning rate of Adam optimizer")
    parser.add_argument('--chunk_size',type=int,default=64,help="chunk size of the model")
    args = parser.parse_args()
    return args



def initial_wandb(args):
    config = {
        "lr":args.lr,
        "epochs":args.epochs,
        "batch_size":args.batch_size,
        "chunk_size":args.chunk_size,
        # "max_length":args.max_length,
        "cuda cores":torch.cuda.device_count(),
        "device name":{i:torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
    }
    wandb.init(project=args.job_name,config=config)

# args = get_args()
# initial_wandb(args)
# a=1

def reduce_mean(tensor, nprocs,device):  # 用于平均所有gpu上的运行结果，比如loss
    if not isinstance(tensor,torch.Tensor):
        tensor = torch.as_tensor(tensor,device=device)
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def ddp_main(args):
    # Load the data
    

    world_size = torch.cuda.device_count()
    local_rank = args.local_rank
    # local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl',world_size=world_size, rank=local_rank)  # nccl的后端通信方式
    # device = torch.device("cuda", local_rank)

    print("start loading parameters...")
    jsonl_file = args.data_jsonl
    split_file = args.split_json
    dataset = StructureDataset(jsonl_file=jsonl_file) # total dataset of the pdb files
    dataset_indices = {d['name']:i for i,d in enumerate(dataset)} # 每个名字对应idx
    with open(f"{split_file}","r") as f:
        dataset_splits = json.load(f)
    print(f"{local_rank} start loading data...")

    train_set0, validation_set0, test_set0 = [
        Subset(dataset, [dataset_indices[chain_name] for chain_name in dataset_splits[key]
        if chain_name in dataset_indices]) for key in ['train', 'validation', 'test']] 
    train_set1, validation_set1, test_set1 = [
        ClusteredDataset(d) for d in [train_set0, validation_set0, test_set0]]

    print(f"ClusteredTraining:{len(train_set1)}, ClusteredValidation:{len(validation_set1)}, ClusteredTest:{len(test_set1)}")
    
    print(f"Rank {local_rank} start loading data...")
    train(args, train_set1, validation_set1)
 

def train(args, train_set, validation_set):
    validation_loader = DataLoader(
        dataset=validation_set, 
        batch_size=args.batch_size, 
        collate_fn=batch_collate_function)
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=False, 
        sampler=train_sampler, 
        collate_fn=batch_collate_function)
    device = torch.device(f"cuda:{args.local_rank}")
    model_path = args.parameters
    torch.hub.set_dir("/pubhome/bozhang/.cache/torch/hub/")
    model = TPFold.load_model(chunk_size=64)
    # model = TPFold._load_model("/pubhome/bozhang/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt") # 从张博那取模型参数？esmfold_3B_v1
    # model = TPFold._load_model(chunk_size=args.chunk_size)
    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank],
        output_device=args.local_rank, find_unused_parameters=True)
    params_1x, params_10x = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in ['tmbed2s.0.weight', 'tmbed2s.2.weight',
                'tmbed2z.dimensionup.weight','tmbed2z.proj.weight','tmbed2z.o_proj.weight','tmbed2z.o_proj.bias']:
                params_10x.append(param)
            else:
                params_1x.append(param)
    optimizer = optim.Adam([{'params': params_1x},
            {'params': params_10x, 'lr': args.lr * 10}], lr=args.lr)
    if args.local_rank == 0:
        print("start training...")
    # initial_wandb(args)
    dist.barrier() 
    best_val_loss = float('inf')
    best_model = None
    best_model_idx = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        # Training epoch
        train_sampler.set_epoch(epoch)  # 这句莫忘，否则相当于没有shuffle数据
        ddp_model.train()
        for iteration, batch in enumerate(train_loader):
            # move train data to different gpu
            for key in batch:
                batch[key] = batch[key].cuda(args.local_rank)
            target_frames = Rigid.from_3_points(batch['C_pos'], batch['CA_pos'], batch['N_pos'])
            optimizer.zero_grad()
            output_dict = model(aa=batch['seq'], mask=batch['mask'], residx=batch['residx'])
            fape_loss = torch.mean(compute_fape(
                            pred_frames=output_dict['target_frames'],
                            target_frames=target_frames,
                            frames_mask=output_dict['frames_mask'],
                            pred_positions=output_dict['backbone_positions'],
                            target_positions=batch['bb_pos'],
                            positions_mask=output_dict['backbone_atoms_mask'],
                            length_scale=10,
                        ))
            loss = fape_loss
            loss.backward()
            optimizer.step()
            
            # 记录
            loss = reduce_mean(loss, dist.get_world_size(),device)
            if args.local_rank == 0:
                metric = {'fape_loss': fape_loss.item(),
                "ave_loss": loss.item(), "epoch": epoch}
                # wandb.log(metric)
            
            
        param_state_dict = ddp_model.module.state_dict()
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': param_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(args.save_folder,f"epoch{epoch}.pt"))
        ddp_model.eval()
        with torch.no_grad():
            val_loss = 0
            val_fape = 0 
            val_seq = 0
            for _, batch in enumerate(validation_loader):
                target_frames = Rigid.from_3_points(batch['C_pos'], batch['CA_pos'], batch['N_pos'])
                output_dict = model(aa=batch['seq'], mask=batch['mask'], residx=batch['residx'])
                fape_loss = torch.mean(
                    compute_fape(
                            pred_frames=output_dict['target_frames'],
                            target_frames=target_frames,
                            frames_mask=output_dict['frames_mask'],
                            pred_positions=output_dict['backbone_positions'],
                            target_positions=batch['bb_pos'],
                            positions_mask=output_dict['backbone_atoms_mask'],
                            length_scale=10,
                        ))
                # val_loss += loss.item()
                # val_seq += loss_seq.item()
                val_fape += fape_loss.item()
            # val_loss /= len(loader_validation)
            # val_seq /= len(loader_validation)
            val_fape /= len(validation_loader)
            metric = {'VAL/fape': val_fape, 
            # 'VAL/ppl': np.exp(val_seq.cpu().data.numpy()), 
            # "VAL/loss":val_loss
            }
            wandb.log(metric)
            if best_val_loss > val_loss:
                best_val_loss = val_loss
args = get_args()
# ddp_main(args)
world_size =2
local_rank = args.local_rank
# local_rank = int(os.environ['LOCAL_RANK'])
dist.init_process_group(backend='nccl',world_size=world_size, rank=local_rank)  # nccl的后端通信方式
torch.cuda.set_device(local_rank)
if __name__ == "__main__":
    args = get_args()
    ddp_main(args)





