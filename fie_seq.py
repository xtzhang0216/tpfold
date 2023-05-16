import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8"
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

from data import StructureDataset, ClusteredDataset_inturn, batch_collate_function
import TPFold 
import utils
import torch.distributed as dist
import torch.optim as optim
from openfold.utils.rigid_utils import Rigid
from openfold.utils.loss import compute_fape
import noam_opt
parser = argparse.ArgumentParser()


parser.add_argument('--shuffle', type=float, default=0., help='Shuffle fraction')
parser.add_argument('--data_jsonl', type=str, default="/pubhome/bozhang/data/tmpnn_v8.jsonl", help='Path for the jsonl data')
parser.add_argument('--split_json', type=str, default="/pubhome/bozhang/data/tmpnn_v8.json",help='Path for the split json file')
parser.add_argument('--output_folder',type=str,default="/pubhome/xtzhang/result/output/",help="output folder for the log files and model parameters")
parser.add_argument('--save_folder',type=str,default="/pubhome/xtzhang/result/save/",help="output folder for the model parameters")
parser.add_argument('--description',type=str,help="description the model information into wandb")
parser.add_argument('--job_name',type=str,default="test",help="jobname of the wandb dashboard")
parser.add_argument('--num_tags',type=int,default=6,help="num tags for the sequence")
parser.add_argument('--epochs',type=int,default=5,help="epochs to train the model")
parser.add_argument('--batch_size',type=int,default=1,help="batch size tokens")
parser.add_argument('--max_length',type=int,default=800,help="max length of the training sequence")
parser.add_argument('--max_tokens',type=int,default=1000,help="max length of the training sequence")
parser.add_argument('--mask',type=float,default=1.0,help="mask fractions into input sequences")
parser.add_argument("--local_rank", help="local device ID", type=int) 
parser.add_argument('--checkpoint_path',type=str,default="./result/save/epoch0.pt" , help="parameters path")
parser.add_argument('--lr',type=float,default=5e-4, help="learning rate of Adam optimizer")
parser.add_argument('--chunk_size',type=int,default=64,help="chunk size of the model")
parser.add_argument('--world_size',type=int,default=2,help="world_size")
parser.add_argument('--pattern',type=str,default="withseq",help="mode")
parser.add_argument('--add_tmbed',type=bool,default=False,help="wheter addtmbed")
parser.add_argument('--RESUME',type=bool,default=False,help="whether use checkpoint")


args = parser.parse_args()




def initial_wandb(args):
    config = {
        "lr":args.lr,
        "epochs":args.epochs,
        "batch_size":args.batch_size,
        "chunk_size":args.chunk_size,
        "num_tags":args.num_tags,
        "max_length":args.max_length,
        "max_tokens":args.max_length,
        "cuda_use":args.world_size,
        "device name":{i:torch.cuda.get_device_name(i) for i in range(args.world_size)}
    }
    wandb.init(project=args.job_name,config=config)

# args = get_args()
# initial_wandb(args)
# a=1

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
    # local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl',world_size=args.world_size, rank=local_rank)  # nccl的后端通信方式
    # device = torch. device("cuda", local_rank)

    print(f"Rank {local_rank} start loading data...")
    jsonl_file = args.data_jsonl
    split_file = args.split_json
    dataset = StructureDataset(jsonl_file=jsonl_file, max_length=args.max_length) # total dataset of the pdb files
    dataset_indices = {d['name']:i for i,d in enumerate(dataset)} # 每个名字对应idx
    with open(f"{split_file}","r") as f:
        dataset_splits = json.load(f)
    # print(f"{local_rank} start loading data...")

    train_set0, validation_set0, test_set0 = [
        Subset(dataset, [dataset_indices[chain_name] for chain_name in dataset_splits[key]
        if chain_name in dataset_indices]) for key in ['train', 'validation', 'test']] 
    train_set1, validation_set1, test_set1 = [
        ClusteredDataset_inturn(d, max_tokens=args.max_tokens, reverse=True) for d in [train_set0, validation_set0, test_set0]]
    if args.local_rank == 0:
        print(f"ClusteredTraining:{len(train_set1)}, ClusteredValidation:{len(validation_set1)}, ClusteredTest:{len(test_set1)}")
    
    
    run(args, train_set1, validation_set1, test_set1)
 
def run(args, train_set, validation_set, test_set):
    train_sampler = DistributedSampler(train_set)
    val_sampler = DistributedSampler(validation_set)
    test_sampler = DistributedSampler(test_set)
    validation_loader = DataLoader(
        dataset=validation_set, 
        batch_size=args.batch_size, 
        collate_fn=batch_collate_function,
        shuffle=False, 
        sampler=val_sampler,)
    test_loader = DataLoader(
        dataset=test_set, 
        batch_size=args.batch_size, 
        collate_fn=batch_collate_function,
        shuffle=False, 
        sampler=test_sampler,)  

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=False, 
        sampler=train_sampler, 
        collate_fn=batch_collate_function)

    device = torch.device(f"cuda:{args.local_rank}")


    if args.local_rank == 0:
        print(f"{args.local_rank} start loading model...")
    torch.hub.set_dir("/pubhome/bozhang/.cache/torch/hub/")
    model = TPFold.load_model(chunk_size=args.chunk_size, pattern=args.pattern, num_tags=args.num_tags, add_tmbed=args.add_tmbed)
    # model = TPFold._load_model("/pubhome/bozhang/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt") # 从张博那取模型参数？esmfold_3B_v1
    # model = TPFold._load_model(chunk_size=args.chunk_size)
    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank],
        output_device=args.local_rank, find_unused_parameters=True)

    params_1x, params_2x = [], []
    for name, param in ddp_model.module.named_parameters():
        if param.requires_grad:
            # if name in ['tmbed2s.0.weight', 'tmbed2s.2.weight',
            #     'tmbed2z.dimensionup.weight','tmbed2z.proj.weight','tmbed2z.o_proj.weight','tmbed2z.o_proj.bias']:
            if name.startswith('tm_s') or name.startswith('trunk.cctop'):
                # print(name)
                params_2x.append(param)
            else:
                params_1x.append(param)
    optimizer = optim.Adam([{'params': params_1x},
            {'params': params_2x, 'lr': args.lr * 2}], lr=args.lr)

    if args.RESUME:
        path_checkpoint = args.checkpoint_path # 断点路径
        checkpoint = torch.load(path_checkpoint, map_location=torch.device('cpu'))
        ddp_model.module.load_state_dict(checkpoint['model_state_dict'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
    else:
        start_epoch = -1

    
    initial_wandb(args)
    wandb.watch(ddp_model.module, log="all", log_freq=10)
    dist.barrier() 
    # best_val_loss = float('inf')
    # best_model = None
    # best_model_idx = 0
    start_time = time.time()
    print(f"{args.local_rank}start training...\n")


    for epoch in range(start_epoch+1, args.epochs):
        # Training epoch
        train_sampler.set_epoch(epoch)  # 这句莫忘，否则相当于没有shuffle数据
        ddp_model.train()
        total_loss = 0.
        total_loss_crf = 0.
        total_loss_fape = 0.
        total_acc_cctop = 0.
        total_weights = 0.
        log_interval = 200
        for iteration, batch in enumerate(train_loader):
            # move train data to different gpu
            for key in batch:
                batch[key] = batch[key].cuda(args.local_rank)
            target_frames = Rigid.from_3_points(batch['C_pos'], batch['CA_pos'], batch['N_pos'])
            S, C, mask = batch["seq"], batch["cctop"], batch["mask"]
            optimizer.zero_grad()
            output_dict = ddp_model(aa=batch['seq'], mask=batch['mask'], residx=batch['residx'])
            loss_fape = torch.mean(compute_fape(
                            pred_frames=output_dict['pred_frames'],
                            target_frames=target_frames,
                            frames_mask=output_dict['frame_mask'],
                            pred_positions=output_dict['backbone_positions'],
                            target_positions=batch['bb_pos'],
                            positions_mask=output_dict['backbone_atoms_mask'],
                            length_scale=10,
                        ))
            logits_cctop = output_dict['logits_cctop']
            # print(output_dict['logits_cctop'].shape, batch['cctop'].shape, batch['mask'].shape)
            loss_crf = ddp_model.module.neg_loss_crf(output_dict['logits_cctop'], batch['cctop'], batch['mask'])
            
            loss = loss_fape + 0.2*loss_crf
            # loss.requires_grad_(True)  
            loss.backward()
            # print("XXX Capturing:", torch.cuda.is_current_stream_capturing())
            optimizer.step()
            lengths = batch["length"]
            num_tokens = (torch.sum(lengths)).item()
            # 手工加上padding
            bag_list = ddp_model.module.decode_crf(logits_cctop, mask)
            for i in range(len(bag_list)):
                if len(bag_list[i]) != S.size(1):
                    bag_list[i] += [0 for _ in range(S.size(1)-len(bag_list[i]))]
            cctop_train = torch.tensor(bag_list,dtype=torch.long)
            cctop_train = cctop_train.cuda(args.local_rank)
            hit_train = torch.sum((cctop_train == C) * mask)
           
            dist.barrier() 
            loss_crf_mean = reduce_mean(loss_crf, dist.get_world_size(), device)
            loss_fape_mean =  reduce_mean(loss_fape, dist.get_world_size(), device)
            loss_mean = reduce_mean(loss, dist.get_world_size(), device)
            acc_mean = reduce_sum(hit_train, device) / reduce_sum(num_tokens, device)

            # 记录
            # loss = reduce_mean(loss, dist.get_world_size(),device)
            if args.local_rank == 0:
                metric = {
                'TRAIN/loss_crf': loss_crf_mean.item(),
                'TRAIN/loss_fape':loss_fape_mean.item(),
                "TRAIN/loss": loss_mean.item(), 
                "TRAIN/acc": acc_mean.item(),
                "epoch": epoch}
                wandb.log(metric)
                print(f'| epoch {epoch:3d} | {iteration:5d}/{len(train_loader):5d} batches | ')
                # print(f'')
            total_loss += loss_mean.item()
            total_loss_crf += loss_crf_mean.item()
            total_loss_fape += loss_fape_mean.item()
            total_acc_cctop += acc_mean.item()

            if iteration % log_interval == 0 and iteration > 0 and args.local_rank == 0:
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_loss_crf = total_loss_crf / log_interval
                # cur_loss_ppl = np.exp(cur_loss_seq)
                cur_loss_fape = total_loss_fape / log_interval
                cur_acc_cctop = total_acc_cctop / log_interval
                # cur_loss_ppl = np.exp(cur_loss_seq)
                print(f'| epoch {epoch:3d} | {iteration:5d}/{len(train_loader):5d} batches | '
                    f'| ms/batch {ms_per_batch:5.2f} | loss {cur_loss:5.2f} | '
                    # f'ppl {cur_loss_ppl:5.2f}|'
                    f'loss_crf {cur_loss_crf} | '
                    f'loss_fape  {cur_loss_fape:5.2f} '
                    f'| acc_cctop{cur_acc_cctop}')
                print(f"bag_list:{bag_list}")
                wandb.log({
                    "TRAIN/loss_fape_xxxsteps": cur_loss_fape,
                    "TRAIN/cur_loss_crf_xxxsteps": cur_loss_crf,
                    "TRAIN/cur_loss_xxxsteps": cur_loss,
                    "TRAIN/cur_acc_cctop_xxxsteps": cur_acc_cctop,
                })
                total_loss = 0.
                total_loss_crf = 0.
                total_loss_fape = 0.
                total_acc_cctop = 0.
                start_time = time.time()
            
        param_state_dict = ddp_model.module.state_dict()
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': param_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(args.save_folder,f"noseq_predictcctop{epoch}.pt"))
        # if args.local_rank == 0:
        ddp_model.eval()
        with torch.no_grad():
            total_loss = 0.
            total_loss_crf = 0.
            total_loss_fape = 0.
            total_acc = 0.


            for _, batch in enumerate(validation_loader):
                for key in batch:
                    batch[key] = batch[key].cuda(args.local_rank)
                S, C, mask, lengths = batch["seq"], batch["cctop"], batch["mask"], batch["length"]                    
                num_tokens = (torch.sum(lengths)).item()
                target_frames = Rigid.from_3_points(batch['C_pos'], batch['CA_pos'], batch['N_pos'])
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
                logits_cctop=output_dict['logits_cctop']                    
                loss_crf = ddp_model.module.neg_loss_crf(logits_cctop,C,mask)
                loss = loss_fape + 0.2*loss_crf
                bag_list = ddp_model.module.decode_crf(logits_cctop,mask)
                for i in range(len(bag_list)):
                    if len(bag_list[i]) != S.size(1):
                        bag_list[i] += [0 for _ in range(S.size(1)-len(bag_list[i]))]
                cctop_validation = torch.tensor(bag_list,dtype=torch.long,device=device)
                hit_val = torch.sum((cctop_validation == C) * mask)                    
                total_loss += loss.item()
                total_loss_crf += loss_crf.item()
                total_loss_fape += loss_fape.item()
                total_acc += hit_val / num_tokens
                # print(args.local_rank, hit_val, num_tokens, total_acc)

            dist.barrier()
            val_loss = reduce_sum(total_loss, device) / len(validation_set)
            val_loss_crf = reduce_sum(total_loss_crf, device) / len(validation_set)
            # cur_loss_ppl = np.exp(cur_loss_seq)
            val_loss_fape = reduce_sum(total_loss_fape, device) / len(validation_set)
            val_acc_cctop = reduce_sum(total_acc, device) / len(validation_set)
            metric = {
            'VAL/loss_crf': val_loss_crf.item(),
            'VAL/loss_fape': val_loss_fape.item(),
            "VAL/loss": val_loss.item(), 
            "VAL/acc":val_acc_cctop
            }
            
            if args.local_rank == 0 :
                wandb.log(metric)
            # if best_val_loss > val_loss and args.local_rank == 0:
            #     best_val_loss = val_loss
            #     best_model = model
            #     best_model_idx = epoch
            #     param_state_dict = ddp_model.module.state_dict()
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': param_state_dict,
            #         'optimizer_state_dict': optimizer.state_dict()
            #     }, os.path.join(args.save_folder,f"epoch{epoch}.pt"))
    
    # if args.local_rank == 0:
    ddp_model.eval()
    with torch.no_grad():
        total_loss = 0.
        total_loss_crf = 0.
        total_loss_fape = 0.
        total_acc_cctop = 0.

        for _, batch in enumerate(test_loader):
            for key in batch:
                    batch[key] = batch[key].cuda(args.local_rank)
            S, C, mask, lengths = batch["seq"], batch["cctop"], batch["mask"],  batch["length"]                   

            target_frames = Rigid.from_3_points(batch['C_pos'], batch['CA_pos'], batch['N_pos'])
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
            logits_cctop=output_dict['logits_cctop']                    
            loss_crf = ddp_model.module.neg_loss_crf(logits_cctop,C,mask)
            loss = loss_fape + 0.2*loss_crf
            bag_list = ddp_model.module.decode_crf(logits_cctop,mask)
            for i in range(len(bag_list)):
                if len(bag_list[i]) != S.size(1):
                    bag_list[i] += [0 for _ in range(S.size(1)-len(bag_list[i]))]
            cctop_validation = torch.tensor(bag_list,dtype=torch.long,device=device)
            hit_test = torch.sum((cctop_validation == C) * mask)   
            total_loss += loss.item()
            total_loss_crf += loss_crf.item()
            total_loss_fape += loss_fape.item()
            total_acc += hit_test / num_tokens
        
        dist.barrier()
        test_loss = reduce_sum(total_loss, device) / len(validation_set)
        test_loss_crf = reduce_sum(total_loss_crf, device) / len(validation_set)
        # cur_loss_ppl = np.exp(cur_loss_seq)
        test_loss_fape = reduce_sum(total_loss_fape, device) / len(validation_set)
        test_acc_cctop = reduce_sum(total_acc, device) / len(validation_set)
    print(f"Fape\t{test_loss_fape :.4f}\nLoss\t{test_loss :.4f}\ncrf\t{test_loss_crf}\nacc\t{test_acc_cctop}")
    # print(f"Perplexity\tTest{np.exp(test_seq.cpu().data.numpy()) :.4f}\nFape\t{test_fape :.4f}\nLoss\t{test_loss :.4f}")

# # ddp_main(args)
# world_size =2
# local_rank = args.local_rank
# # local_rank = int(os.environ['LOCAL_RANK'])
# dist.init_process_group(backend='nccl',world_size=world_size, rank=local_rank)  # nccl的后端通信方式
# torch.cuda.set_device(local_rank)
if __name__ == "__main__":
    # args = get_args()
    ddp_main(args)





