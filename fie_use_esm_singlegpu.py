import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8"
import typing as T
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
from data import Crop_StructureDataset, batch_collate_function_nocluster
 
from in_plm_fold import load_model 
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
parser.add_argument('--job_name',type=str,default="200aa_test",help="jobname of the wandb dashboard")
parser.add_argument('--num_tags',type=int,default=6,help="num tags for the sequence")
parser.add_argument('--epochs',type=int,default=5,help="epochs to train the model")
parser.add_argument('--batch_size',type=int,default=4,help="batch size tokens")
parser.add_argument('--max_length',type=int,default=800,help="max length of the training sequence")
parser.add_argument('--max_tokens',type=int,default=1000,help="max length of the training sequence")
parser.add_argument('--mask',type=float,default=1.0,help="mask fractions into input sequences")
parser.add_argument("--local_rank", default=0, help="local device ID", type=int) 
parser.add_argument('--parameters',type=str,default="/pubhome/bozhang/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt", help="parameters path")
parser.add_argument('--lr',type=float,default=5e-4, help="learning rate of Adam optimizer")
parser.add_argument('--chunk_size',type=int,default=4,help="chunk size of the model")
parser.add_argument('--world_size',type=int,default=2,help="world_size")
parser.add_argument('--pattern',type=str,default="no",help="mode")
parser.add_argument('--crop_size',type=int,default=600, help="watch")
parser.add_argument('--accumulation_steps',type=int,default=64, help="watch")


args = parser.parse_args()


def initial_wandb(args):
    config = {
        "lr":args.lr,
        "epochs":args.epochs,
        "batch_size":args.batch_size,
        "chunk_size":args.chunk_size,
        "max_length":args.max_length,
        "max_tokens":args.max_length,
    }
    wandb.init(project=args.job_name,config=config)





def main(args):
    # Load the data
    

    # world_size = torch.cuda.device_count()
    # local_rank = int(os.environ['LOCAL_RANK'])
    # device = torch. device("cuda", local_rank)
    device = torch.device("cuda:0")
    print(f"device:{device}")
    print(f"start loading data...")
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
    # train_set0, validation_set0, test_set0 = dataset,dataset,dataset
    
    run(args, train_set0, validation_set0, test_set0)
 

def run(args, train_set, validation_set, test_set):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    esm_dict = torch.load("/pubhome/xtzhang/myesm/esm2_output.pt")
    # for i in train_set
    validation_loader = DataLoader(
        dataset=validation_set, 
        batch_size=args.batch_size, 
        collate_fn=batch_collate_function_nocluster,
        shuffle=False,)
    test_loader = DataLoader(
        dataset=test_set, 
        batch_size=args.batch_size, 
        collate_fn=batch_collate_function_nocluster,
        shuffle=False, )  

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=False, 
        collate_fn=batch_collate_function_nocluster)

    print(" start loading model...")
    model_path = args.parameters
    torch.hub.set_dir("/pubhome/bozhang/.cache/torch/hub/")
    model = load_model(chunk_size=64, 
        pattern=args.pattern,
        # esm2_path="/pubhome/xtzhang/plm/esm2_t36_3B_UR50D_finetuned.pt",
        # ft_path="/pubhome/xtzhang/plm/token_cctop/epoch14.pt"
        )
    # model_data = torch.load(str(model_path), map_location="cuda:0") #读取一个pickle文件为一个dict
    # cfg = model_data["cfg"]["model"]
    # model = esmfold.ESMFold(pattern=args.pattern, esmfold_config=cfg) # make an instance
    # model_state = model_data["model"]
    # model.load_state_dict(model_state, strict=False)
    # model.set_chunk_size(args.chunk_size)
    model = model.to(device)

    params_1x, params_2x = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # if name in ['tmbed2s.0.weight', 'tmbed2s.2.weight',
            #     'tmbed2z.dimensionup.weight','tmbed2z.proj.weight','tmbed2z.o_proj.weight','tmbed2z.o_proj.bias']:
            if name.startswith('tm'):
                params_2x.append(param)
            else:
                params_1x.append(param)
    optimizer = optim.Adam([{'params': params_1x},
            {'params': params_2x, 'lr': args.lr * 2}], lr=args.lr)

    initial_wandb(args)
    wandb.watch(model, log="all", log_freq=50)
    best_val_loss = float('inf')
    best_model = None
    best_model_idx = 0
    start_time = time.time()
    print("start training...\n")

    for epoch in range(args.epochs):
        # Training epoch
        model.train()
        total_loss = 0.
        # total_loss_seq = 0.
        total_loss_fape = 0.
        total_weights = 0.

        log_interval = 500
        for iteration, batch in enumerate(train_loader):
            # move train data to different gpu
            for key in batch:
                if key != 'name':
                    batch[key] = batch[key].cuda(args.local_rank)
            name, C_pos, CA_pos, N_pos, seq, mask, residx, bb_pos = batch['name'], batch['C_pos'], batch['CA_pos'], batch['N_pos'],batch['seq'], batch['mask'], batch['residx'], batch['bb_pos']
            # C_pos, CA_pos, N_pos, seq, mask, residx, bb_pos = set_grade(C_pos, CA_pos, N_pos, seq, mask, residx, bb_pos)
            # target_frames = Rigid.from_3_points(C_pos, CA_pos, N_pos)
            target_frames = get_bb_frames(torch.stack((N_pos, CA_pos, C_pos), dim=-2))
            # print(N_pos.shape)
            # print(torch.stack((N_pos, CA_pos, C_pos), dim=-2).shape)
            # 如果输入维度有batch维，则先降维再padding
            esm_s = utils.CoordBatchConverter.collate_dense_tensors([esm_dict[n].squeeze(0) for n in name ],0.0)
            esm_s = esm_s.to(device)
            if esm_s.shape[1] > args.crop_size:
                esm_s = esm_s[:, 0:args.crop_size, :]
            output_dict = model(esm_s=esm_s, aa=seq, mask=mask, residx=residx)
            loss_fape = torch.mean(compute_fape(
                            pred_frames=output_dict['pred_frames'],
                            target_frames=target_frames,
                            frames_mask=output_dict['frame_mask'],
                            pred_positions=output_dict['backbone_positions'],
                            target_positions=bb_pos,
                            positions_mask=output_dict['backbone_atoms_mask'],
                            length_scale=10,
                        ))
            # wandb.watch(ddp_model, log="all", log_freq=1)
            # loss.requires_grad_(True)  
            loss_fape.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            if (iteration+1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # 记录
            # print(f"lengths, cuda:{args.local_rank}: {lengths}")
            # print(f"lossfape, cuda:{args.local_rank}: {loss_fape}")


            metric = {'TRAIN/loss_fape': (loss_fape).item(),
            # "TRAIN/loss": loss.item(),
            "epoch": epoch}
            wandb.log(metric)
            print(f'TRAIN: | epoch {epoch:3d} | {iteration:5d}/{len(train_loader):5d} batches | ')
        
            total_loss_fape += loss_fape.item()
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
        if (iteration+1) % args.accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
        # param_state_dict = ddp_model.module.state_dict()
        # torch.save({
        #             'epoch': epoch,
        #             'model_state_dict': param_state_dict,
        #             'optimizer_state_dict': optimizer.state_dict()
        #         }, os.path.join(args.save_folder,f"epoch{epoch}.pt"))
        # if args.local_rank == 0:
        model.eval()
        with torch.no_grad():
            # total_weights = 0.
            total_loss_fape = 0.
            for iteration, batch in enumerate(validation_loader):
                for key in batch:
                    batch[key] = batch[key].cuda(args.local_rank)
                mask = batch['mask']
                target_frames = Rigid.from_3_points(batch['C_pos'], batch['CA_pos'], batch['N_pos'])
                output_dict = model(aa=batch['seq'], mask=batch['mask'], residx=batch['residx'])
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
                total_weights += torch.sum(mask)
                # total_loss_seq += loss_seq.item()
                total_loss_fape += loss_fape.item()
                print(f'VAL: | epoch {epoch:3d} | {iteration:5d}/{len(validation_loader):5d} batches | ')
            
            val_loss_fape = total_loss_fape / total_weights
            metric = {'VAL/fape': val_loss_fape, 
            # 'VAL/ppl': np.exp(val_seq.cpu().data.numpy()), 
            # "VAL/loss":val_loss
            }
            
            if args.local_rank == 0 :
                wandb.log(metric)
            if best_val_loss > val_loss_fape and args.local_rank == 0:
                best_val_loss = val_loss_fape
                best_model = model
                best_model_idx = epoch
                param_state_dict = model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': param_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(args.save_folder,f"epoch{epoch}.pt"))

    if args.local_rank == 0:
        model.eval()
        with torch.no_grad():
            total_weights = 0.
            total_loss_fape = 0.
            for iteration, batch in enumerate(test_loader):
                for key in batch:
                        batch[key] = batch[key].cuda(args.local_rank)
                mask = batch["mask"]
                target_frames = Rigid.from_3_points(batch['C_pos'], batch['CA_pos'], batch['N_pos'])
                output_dict = model(aa=batch['seq'], mask=batch['mask'], residx=batch['residx'])
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
                # # seq loss (nll)
                # log_pred = output['log_softmax_aa']
                # padding_mask = batch['padding_mask']
                # loss_seq_token, loss_seq = utils.loss_nll(batch['seq'], log_pred, padding_mask)

                # structure loss (fape)
                # B,L = output['positions'].shape[:2]
                # pred_position = output['positions'].reshape(B, -1, 3)
                # target_position = batch['coord'][:,:,:3,:].reshape(B, -1 ,3)
                # position_mask = torch.ones_like(target_position[...,0])
                # loss_fape = torch.mean(utils.compute_fape(output['pred_frames'],output['target_frames'],batch['padding_mask'],pred_position,target_position,position_mask,10.0))
                # loss = 2*fapse + 0.2*cse copy from DeepMind AlphaFold2 loss function
                loss =  loss_fape            
                total_loss_fape += loss_fape.item()
                total_weights += torch.sum(mask)
                print(f'Test: |  {iteration:5d}/{len(test_loader):5d} batches | ')

            test_loss_fape = total_loss_fape / total_weights

        print(f"Fape\t{test_loss_fape :.4f}")
        # print(f"Perplexity\tTest{np.exp(test_seq.cpu().data.numpy()) :.4f}\nFape\t{test_fape :.4f}\nLoss\t{test_loss :.4f}")

# # ddp_main(args)
# world_size =2
# local_rank = args.local_rank
# # local_rank = int(os.environ['LOCAL_RANK'])
# dist.init_process_group(backend='nccl',world_size=world_size, rank=local_rank)  # nccl的后端通信方式
# torch.cuda.set_device(local_rank)
if __name__ == "__main__":
    # args = get_args()
    main(args)





