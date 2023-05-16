import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8"
import typing as T

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
from data import Crop_StructureDataset, batch_collate_function_nocluster
 
import TPFold 
from utils import get_bb_frames
import torch.distributed as dist
import torch.optim as optim
from openfold.utils.rigid_utils import Rigid
from openfold.utils.loss import compute_fape
import noam_opt
parser = argparse.ArgumentParser()


parser.add_argument('--shuffle', type=float, default=0., help='Shuffle fraction')
parser.add_argument('--data_jsonl', type=str, default="/pubhome/xtzhang/data/cctop.jsonl", help='Path for the jsonl data')
parser.add_argument('--split_json', type=str, default="/pubhome/bozhang/data/tmpnn_v8.json",help='Path for the split json file')
parser.add_argument('--output_folder',type=str,default="/pubhome/xtzhang/result/output/",help="output folder for the log files and model parameters")
parser.add_argument('--save_folder',type=str,default="/pubhome/xtzhang/result/save/",help="output folder for the model parameters")
parser.add_argument('--description',type=str,help="description the model information into wandb")
parser.add_argument('--project_name',type=str,default="800aa_noseq_nocctop",help="jobname of the wandb dashboard")
parser.add_argument('--run_name',type=str,default="800aa_noseq_nocctop",help="jobname of the wandb dashboard")
parser.add_argument('--num_tags',type=int,default=6,help="num tags for the sequence")
parser.add_argument('--epochs',type=int,default=5,help="epochs to train the model")
parser.add_argument('--batch_size',type=int,default=1,help="batch size tokens")
parser.add_argument('--max_length',type=int,default=800,help="max length of the training sequence")
parser.add_argument('--max_tokens',type=int,default=800,help="max length of the training sequence")
parser.add_argument('--mask',type=float,default=1.0,help="mask fractions into input sequences")
parser.add_argument("--local_rank", help="local device ID", type=int) 
parser.add_argument('--parameters',type=str,default="/pubhome/bozhang/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt", help="parameters path")
parser.add_argument('--lr',type=float,default=5e-4, help="learning rate of Adam optimizer")
parser.add_argument('--chunk_size',type=int,default=64,help="chunk size of the model")
parser.add_argument('--world_size',type=int,default=4,help="world_size")
parser.add_argument('--pattern',type=str,default="no",help="mode")
parser.add_argument('--add_tmbed',type=bool,default=False, help="whether addtmbed")
parser.add_argument('--watch_freq',type=int,default=500, help="watch")
parser.add_argument('--crop_size',type=int,default=256, help="watch")

args = parser.parse_args()



def initial_wandb(args):
    config = {
        "lr":args.lr,
        "epochs":args.epochs,
        "batch_size":args.batch_size,
        "chunk_size":args.chunk_size,
        "max_length":args.max_length,
        "max_tokens":args.max_length,
        "cuda_use": dist.get_world_size(),
        "device name":{i:torch.cuda.get_device_name(i) for i in range(dist.get_world_size())}
    }
    wandb.init(project=args.project_name, name=args.run_name+str(args.local_rank), config=config)


def reduce_mean(tensor, nprocs, device):  # 用于平均所有gpu上的运行结果，比如loss
    if not isinstance(tensor,torch.Tensor):
        tensor = torch.as_tensor(tensor,device=device)
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def reduce_sum(tensor, device):  # 用于平均所有gpu上的运行结果，比如loss
    if not isinstance(tensor,torch.Tensor):
        tensor = torch.as_tensor(tensor,device=device)
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    # rt /= nprocs
    return rt


def ddp_main(args):
    # Load the data

    # world_size = torch.cuda.device_count()
    local_rank = args.local_rank
    # local_rank = int(os.environ['LO CAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl',world_size=args.world_size, rank=local_rank)  # nccl的后端通信方式
    # device = torch. device("cuda", local_rank)

    print(f"Rank {local_rank} start loading data...")
    jsonl_file = args.data_jsonl
    split_file = args.split_json
    dataset = Crop_StructureDataset(jsonl_file=jsonl_file, max_length=args.max_length, crop_size=args.crop_size) # total dataset of the pdb files
    dataset_indices = {d['name']:i for i,d in enumerate(dataset)} # 每个名字对应idx
    with open(f"{split_file}","r") as f:
        dataset_splits = json.load(f)
    # print(f"{local_rank} start loading data...")

    train_set0, validation_set0, test_set0 = [
        Subset(dataset, [dataset_indices[chain_name] for chain_name in dataset_splits[key]
        if chain_name in dataset_indices]) for key in ['train', 'validation', 'test']] 
    # train_set1, validation_set1, test_set1 = [
    #     ClusteredDataset_inturn(d, max_tokens=args.max_tokens) for d in [train_set0, validation_set0, test_set0]]
    if args.local_rank == 0:
        print(f"ClusteredTraining:{len(train_set0)}, ClusteredValidation:{len(validation_set0)}, ClusteredTest:{len(test_set0)}")
    
    
    run(args, train_set0, validation_set0, test_set0)
 

def run(args, train_set, validation_set, test_set):
    train_sampler = DistributedSampler(train_set)
    val_sampler = DistributedSampler(validation_set)
    test_sampler = DistributedSampler(test_set)
    validation_loader = DataLoader(
        dataset=validation_set, 
        batch_size=args.batch_size, 
        collate_fn=batch_collate_function_nocluster,
        shuffle=False, 
        sampler=val_sampler,)
    test_loader = DataLoader(
        dataset=test_set, 
        batch_size=args.batch_size, 
        collate_fn=batch_collate_function_nocluster,
        shuffle=False, 
        sampler=test_sampler,)  

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=False, 
        sampler=train_sampler, 
        collate_fn=batch_collate_function_nocluster)

    device = torch.device(f"cuda:{args.local_rank}")
    if args.local_rank == 0:
        print(f"{args.local_rank} start loading model...")
    model_path = args.parameters
    # torch.hub.set_dir("/pubhome/bozhang/.cache/torch/hub/")
    model = TPFold.load_model(
        chunk_size=64, 
        add_tmbed = False, 
        pattern=args.pattern,
        # esm2_path="/pubhome/xtzhang/plm/esm2_t36_3B_UR50D_finetuned.pt",
        # ft_path="/pubhome/xtzhang/plm/token_cctop/epoch14.pt"
        )
    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank],
        output_device=args.local_rank, find_unused_parameters=True,
        # static_graph=True,
        )
    params_1x, params_2x = [], []
    for name, param in ddp_model.module.named_parameters():
        if param.requires_grad:
            # if name in ['tmbed2s.0.weight', 'tmbed2s.2.weight',
            #     'tmbed2z.dimensionup.weight','tmbed2z.proj.weight','tmbed2z.o_proj.weight','tmbed2z.o_proj.bias']:
            # print(name)
            if name.startswith('tm_s'):
                params_2x.append(param)
                print(name)
            else:
                params_1x.append(param)
    optimizer = optim.Adam([{'params': params_1x},
            {'params': params_2x, 'lr': args.lr * 2}], lr=args.lr)
    # for name, param in ddp_model.module.named_parameters():
    #  if param.requires_grad:
    #      print(name)

    initial_wandb(args)
    wandb.watch(ddp_model.module, log="all", log_freq=args.watch_freq)
    dist.barrier() 
    best_val_loss = float('inf')
    best_model = None
    best_model_idx = 0
    start_time = time.time()
    print(f"{args.local_rank}start training...\n")

    for epoch in range(args.epochs):
        # Training epoch
        train_sampler.set_epoch(epoch)  # 这句莫忘，否则相当于没有shuffle数据
        ddp_model.train()
        total_loss_fape = 0.
        log_interval = 100
        # print(f"len: {len(train_loader)}")
        for iteration, batch in enumerate(train_loader):
            # move train data to different gpu
            for key in batch:
                batch[key] = batch[key].cuda(args.local_rank)
            C_pos, CA_pos, N_pos, seq, mask, residx, bb_pos = batch['C_pos'], batch['CA_pos'], batch['N_pos'],batch['seq'], batch['mask'], batch['residx'], batch['bb_pos']
            # C_pos, CA_pos, N_pos, seq, mask, residx, bb_pos = set_grade(C_pos, CA_pos, N_pos, seq, mask, residx, bb_pos)
            # target_frames = Rigid.from_3_points(C_pos, CA_pos, N_pos)
            # print(N_pos.shape)
            # print(torch.stack((N_pos, CA_pos, C_pos), dim=-2).shape)
            target_frames = get_bb_frames(torch.stack((N_pos, CA_pos, C_pos), dim=-2))

            optimizer.zero_grad()
            output_dict = ddp_model(aa=seq, mask=mask, residx=residx)
            loss_fape = torch.mean(compute_fape(
                            pred_frames=output_dict['pred_frames'],
                            target_frames=target_frames,
                            frames_mask=output_dict['frame_mask'],
                            pred_positions=output_dict['backbone_positions'],
                            target_positions=bb_pos,
                            positions_mask=output_dict['backbone_atoms_mask'],
                            length_scale=10,
                        ))
            # for n, p in model.named_parameters():
            #     if p.grad is None and p.requires_grad is True:
            #         print('No forward parameters:', n, p.shape)
            # wandb.watch(ddp_model, log="all", log_freq=1)
            # loss.requires_grad_(True)  
            loss_fape.backward()
            optimizer.step()
            
            # 记录
            # print(f"lengths, cuda:{args.local_rank}: {lengths}")
            # print(f"lossfape, cuda:{args.local_rank}: {loss_fape}")
            dist.barrier() 
            loss_mean = reduce_mean(loss_fape, dist.get_world_size(), device)            

            if args.local_rank == 0:
                metric = {'TRAIN/loss_fape': (loss_mean).item(),
                # "TRAIN/loss": loss.item(),
                "epoch": epoch}
                wandb.log(metric)
                print(f'TRAIN: | epoch {epoch:3d} | {iteration:5d}/{len(train_loader):5d} batches | ')
            

            total_loss_fape += loss_mean.item()

            if iteration % log_interval == 0 and iteration > 0 and args.local_rank == 0:
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                # cur_loss = total_loss / total_weights
                cur_loss_fape = total_loss_fape / log_interval
                print(f'| epoch {epoch:3d} | {iteration:5d}/{len(train_loader):5d} batches | '
                    f'| ms/batch {ms_per_batch:5.2f} | '
                    # f' loss {cur_loss:5.2f} | '
                    # f'ppl {cur_loss_ppl:5.2f}|'
                    f'  {cur_loss_fape:5.2f}')
                wandb.log({
                    "TRAIN/loss_fape_100steps": cur_loss_fape
                })
                total_loss_fape = 0.
                start_time = time.time()
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        param_state_dict = ddp_model.module.state_dict()
        torch.save({
                    'epoch': epoch,
                    # 'step': iteration,
                    'model_state_dict': param_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(args.save_folder,f"{args.run_name}epoch{epoch}.pt"))

        # if args.local_rank == 0:
        ddp_model.eval()
        with torch.no_grad():
            total_loss_fape = 0.
            for iteration, batch in enumerate(validation_loader):
                for key in batch:
                    batch[key] = batch[key].cuda(args.local_rank)
                target_frames = get_bb_frames(torch.stack((N_pos, CA_pos, C_pos), dim=-2))
                # target_frames = Rigid.from_3_points(batch['C_pos'], batch['CA_pos'], batch['N_pos'])
                output_dict = ddp_model(aa=batch['seq'], mask=batch['mask'], residx=batch['residx'])
                loss_fape = torch.mean(
                    compute_fape(
                            pred_frames=output_dict['pred_frames'],
                            target_frames=target_frames,
                            frames_mask=output_dict['frame_mask'],
                            pred_positions=output_dict['backbone_positions'],
                            target_positions=batch['bb_pos'],
                            positions_mask=output_dict['backbone_atoms_mask'],
                            length_scale=10,
                        ))
                total_loss_fape += loss_fape.item()
                # print(f"cuda{args.local_rank} loss_fape={loss_fape}\n")
                # print(f"cuda{args.local_rank} total_loss_fape={total_loss_fape}\n")
                
            # if args.local_rank == 0:
            #     print(f'VAL: | epoch {epoch:3d} | {iteration:5d}/{len(validation_loader):5d} batches | ')
            
            dist.barrier()             
            val_loss_fape = reduce_sum(total_loss_fape, device) / len(validation_set)
            
            if args.local_rank == 0 :
                # print(f"val_loss_fape={val_loss_fape}")
                wandb.log({
                    'VAL/fape': val_loss_fape, 
            })

            # if best_val_loss > val_loss_fape and args.local_rank == 0:
            #     best_val_loss = val_loss_fape
            #     best_model = ddp_model
            #     best_model_idx = epoch
            #     param_state_dict = ddp_model.module.state_dict()
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': param_state_dict,
            #         'optimizer_state_dict': optimizer.state_dict()
            #     }, os.path.join(args.save_folder,f"best_epoch{epoch}.pt"))

    # if args.local_rank == 0:
    ddp_model.eval()
    with torch.no_grad():
        total_loss_fape = 0.
        for iteration, batch in enumerate(test_loader):
            for key in batch:
                    batch[key] = batch[key].cuda(args.local_rank)
            target_frames = get_bb_frames(torch.stack((N_pos, CA_pos, C_pos), dim=-2))
            
            # target_frames = Rigid.from_3_points(batch['C_pos'], batch['CA_pos'], batch['N_pos'])
            output_dict = ddp_model(aa=batch['seq'], mask=batch['mask'], residx=batch['residx'])
            loss_fape = torch.mean(
                    compute_fape(
                            pred_frames=output_dict['pred_frames'],
                            target_frames=target_frames,
                            frames_mask=output_dict['frame_mask'],
                            pred_positions=output_dict['backbone_positions'],
                            target_positions=batch['bb_pos'],
                            positions_mask=output_dict['backbone_atoms_mask'],
                            length_scale=10,
                        ))
            total_loss_fape += loss_fape.item()
            if args.local_rank == 0:
                print(f'Test: |  {iteration:5d}/{len(test_loader):5d} batches | ')
            
        dist.barrier()             
        test_loss_fape = reduce_sum(total_loss_fape, device) / len(test_set)


    print(f"Fape\t{test_loss_fape :.4f}")

if __name__ == "__main__":
    # args = get_args()
    ddp_main(args)





