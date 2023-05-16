import torch
from torch.utils.data import Dataset, DataLoader
import TPFold
import json
from torch.utils.data.dataset import Subset

from data import StructureDataset,ClusteredDataset_inturn,batch_collate_function_withname
split_file="/pubhome/bozhang/data/tmpnn_v8.json"
jsonl_file = "/pubhome/bozhang/data/tmpnn_v8.jsonl"
dataset = StructureDataset(jsonl_file=jsonl_file, max_length=8000) # total dataset of the pdb files
# dataset2= ClusteredDataset_inturn(dataset, max_tokens=800)
dataset_indices = {d['name']:i for i,d in enumerate(dataset)} # 每个名字对应idx
with open(f"{split_file}","r") as f:
    dataset_splits = json.load(f)
# print(f"{local_rank} start loading data...")

train_set0, validation_set0, test_set0 = [
    Subset(dataset, [dataset_indices[chain_name] for chain_name in dataset_splits[key]
    if chain_name in dataset_indices]) for key in ['train', 'validation', 'test']] 
model = TPFold.load_model(chunk_size=64, 
        esm2_path="/lustre/gst/xuchunfu/zhangxt/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt", 
        ft_path="/lustre/gst/xuchunfu/zhangxt/checkpoint/token__gpu4_2/epoch6.pt"
        )
device=torch.device('cuda')
cpu=torch.device('cpu')
model=model.to(device)
# train_loader = DataLoader(
#         dataset=dataset2,
#         batch_size=1,
#         shuffle=False, 
#         collate_fn=batch_collate_function_withname)
# for iteration, batch in enumerate(train_loader):
#             # move train data to different gpu
#             for key in batch:
#                 batch[key] = batch[key].cuda(0)
#             C_pos, CA_pos, N_pos, seq, mask, residx, bb_pos = batch['C_pos'], batch['CA_pos'], batch['N_pos'],batch['seq'], batch['mask'], batch['residx'], batch['bb_pos']
#             # C_pos, CA_pos, N_pos, seq, mask, residx, bb_pos = set_grade(C_pos, CA_pos, N_pos, seq, mask, residx, bb_pos)
import pickle
save_path = "/pubhome/xtzhang/myesm/esm2_output.pt"
# with open(save_path, "ab") as f:
output={}
with torch.no_grad():
    for i,pro in enumerate(dataset):
        seq, name, mask, residx = pro['seq'], pro['name'],pro['mask'],pro['residx']
        seq=seq.to(device);mask=mask.to(device);residx=residx.to(device)
        # print(f"cmpute {i} / {len(train_set0)}   length: {seq.shape}")

        seq, mask, residx = seq[None,...], mask[None,...], residx[None,...]
        esm_s = model(aa=seq, mask=mask, residx=residx)
        esm_s=esm_s.to(cpu)
        output[name]=esm_s.squeeze(0)
    # pickle.dump({name: esm_s})
torch.save(output,save_path)
# output_dict = torch.load(save_path)