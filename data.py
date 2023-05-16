from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.nn as nn
import numpy as np
import json
import copy
import random
import utils
# from . import utils
from esm.esmfold.v1.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
    output_to_pdb,
)
import math
from typing import TypeVar, Optional, Iterator
from openfold.utils.rigid_utils import Rigid
import torch.distributed as dist
restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
restype_order = {restype: i for i, restype in enumerate(restypes)}


def get_single_crop_idx(
    num_res, crop_size: int
) -> torch.Tensor:
    num_res = int(num_res)
    if num_res < crop_size:
        return torch.arange(num_res)
    crop_start = int(np.random.randint(0, num_res - crop_size + 1))
    return torch.arange(crop_start, crop_start + crop_size)


class easyset(Dataset):
    def __init__(self, jsonl_file):
        self.data = utils.load_jsonl(jsonl_file)
        for i in self.data:
            if i['name'].startswith("AF"):
                i['name'] += "_A"
            i['cctop'] = i['cctop'].replace("T","S") 
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

class StructureDataset(Dataset):
    def __init__(self,jsonl_file,max_length=500,low_fraction=0.6,high_fraction=0.9):
        dset = utils.load_jsonl(jsonl_file)
        for i in dset:
            if i['name'].startswith("AF"):
                i['name'] += "_A"
            i['cctop'] = i['cctop'].replace("T","S") #替换减少一类
        cctop_code = 'IMOULS'
        self.data = []
        self.discard = {"bad_chars":0,"too_long":0, "not_match":0}
        
        for entry in dset:
            name = entry['name']
            seq = entry['seq']
            cctop = entry['cctop']
            length = torch.tensor([len(seq)],dtype=torch.long)
            # Check  if in alphabet
            bad_chars = set([s for s in seq]).difference(utils.restypes)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    pass
                else:
                    self.discard['too_long'] += 1
                    continue
            else:
                # print(entry['name'], bad_chars, entry['seq'])
                self.discard['bad_chars'] += 1
                continue
            seq = torch.tensor([restype_order[i] for i in seq],dtype=torch.long)
            cctop = torch.tensor([cctop_code.index(i) for i in cctop],dtype=torch.long)
            coords = torch.from_numpy(np.stack(list(entry['coords'].values()),axis=-2))
            coords = coords.to(torch.float32)
            coords = utils.nan_to_num(coords) # remove the nan value
            N_pos = torch.tensor(entry['coords']['N'])
            C_pos = torch.tensor(entry['coords']['C'])
            CA_pos = torch.tensor(entry['coords']['CA'])
            CB_pos = torch.tensor(entry['coords']['CB'])
            
            mask = torch.ones(length.item(), dtype=int)
            residx = torch.arange(length.item(), dtype=int)
            if coords.shape[0] != seq.shape[0] or cctop.shape[0] != seq.shape[0]:
                self.discard['not_match'] += 1
                continue
            self.data.append({
                "name":name,
                "coords":coords,
                "N_pos":N_pos,
                "C_pos":C_pos,
                "CA_pos":CA_pos,
                "CB_pos":CB_pos,
                "seq":seq,
                "cctop":cctop,
                "mask":mask,
                "residx":residx,
                "length":length
            })
        print(f"UNK token:{self.discard['bad_chars']},too long:{self.discard['too_long']}, 'not_match':{self.discard['not_match']}")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

class ClusteredDataset0(Dataset):
    def __init__(self, jsonl_file,max_length=500, batch_size=10000, shuffle=True,
                 collate_fn=lambda x: x, drop_last=False):
        dataset = utils.load_jsonl(jsonl_file)
        for i in dataset:
            if i['name'].startswith("AF"):
                i['name'] += "_A"
            i['cctop'] = i['cctop'].replace("T","S") #替换减少一类
        cctop_code = 'IMOULS'
        self.data = []
        self.discard = {"bad_chars":0,"too_long":0}
        
        for entry in dataset:
            name = entry['name']
            seq = entry['seq']
            cctop = entry['cctop']
            length = torch.tensor([len(seq)],dtype=torch.long)
            # Check if in alphabet
            bad_chars = set([s for s in seq]).difference(utils.restypes)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    pass
                else:
                    self.discard['too_long'] += 1
                    continue
            else:
                # print(entry['name'], bad_chars, entry['seq'])
                self.discard['bad_chars'] += 1
                continue
            seq = torch.tensor([restype_order[i] for i in seq],dtype=torch.long)
            cctop = torch.tensor([cctop_code.index(i) for i in cctop],dtype=torch.long)
            coord = torch.from_numpy(np.stack(list(entry['coords'].values()),axis=-2))
            coord = coord.to(torch.float32)
            coord = utils.nan_to_num(coord) # remove the nan value
            # seq_mask_fraction = torch.tensor([np.random.uniform(low=low_fraction, high=high_fraction),],dtype=torch.float32)
            # seq_mask = []
            # for _ in range(len(seq)):
            #     if np.random.random() < seq_mask_fraction:
            #         seq_mask.append(False)
            #     else:
            #         seq_mask.append(True)
            # seq_mask = torch.tensor(seq_mask,dtype=torch.bool) # 0.0, mask; 1 unmask
            mask_seq = copy.deepcopy(seq)
            mask_seq[~seq_mask] = 20

            self.data.append({
                "name":name,
                "coord":coord,
                "seq":seq,
                "cctop":cctop,
                # "seq_mask":seq_mask,
                # "mask_seq":mask_seq,
                # "seq_mask_fraction":seq_mask_fraction,
                "length":length
            })
        print(f"UNK token:{self.discard['bad_chars']},too long:{self.discard['too_long']}")

        self.size = len(self.data)
        self.lengths = [dataset[i]['length'] for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths) # 默认原来的dataset有长有短，现在排序之后更方便batch

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix: #遍历dataset的所有样本得到许多小的minibatch
            size = self.lengths[ix] #第ix个样本的长度,也是这个batch里最大的长度(本身sorted_ix就是从小往大排序)
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(self.data[ix])
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0: #最后一个mini batch可能还剩下一点东西
            clusters.append(batch)
        self.clusters = clusters #不同minibatch 组成的 token，但最大只能有6000个氨基酸

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, idx):
        np.random.shuffle(self.clusters)
        return self.clusters[idx]

class ClusteredDataset_noturn(Dataset):
    def __init__(self, dataset, max_tokens=700, shuffle=True,
                drop_last=False): #max_tokens限制最长词元（氨基酸）个数
        self.data = []
        for i in dataset:
           self.data.append(i)
        random.shuffle(self.data)

        self.size = len(self.data)
        self.lengths = [self.data[i]['length'] for i in range(self.size)]
        # self.max_token = max_tokens
        # sorted_ix = np.argsort(self.lengths) # 默认原来的dataset有长有短，现在排序之后更方便batch

        # Cluster into batches of similar sizes
        self.clusters, batch = [], []
        size_ex = 0
        for i in range(self.size): 
            size = self.lengths[i] 
            if size + size_ex <= max_tokens: #如果token数超过已有token，就归到下一个batch里
                batch.append(self.data[i])
                size_ex += size
            else:
                self.clusters.append(batch)
                batch = []
                batch.append(self.data[i])
                size_ex = size

        if len(batch) > 0: #最后一个batch可能还剩下一点东西
            self.clusters.append(batch)
        

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, idx):
        # np.random.shuffle(self.clusters)
        return self.clusters[idx]

class ClusteredDataset_inturn(Dataset):
    def __init__(self, dataset,max_tokens=400, reverse=True, shuffle=True,
                  drop_last=False):
        
        # self.size = len(dataset)
        lengths = [dataset[i]['length'] for i in range(len(dataset))]
        # self.batch_size = batch_size
        sorted_ix = np.argsort(lengths) 

        # Cluster into batches of similar sizes
        self.clusters, batch = [], []
        # for i,ix in enumerate(sorted_ix):
        #     num = max_tokens // lengths[ix]
        #     if 
        #     for j in num:
        #         batch.append(dataset[j])

        batch_max = 0
        for ix in sorted_ix: #遍历dataset的所有样本得到许多小的minibatch
            size = lengths[ix] #第ix个样本的长度,也是这个batch里最大的长度(本身sorted_ix就是从小往大排序)
            if size * (len(batch) + 1) <= max_tokens:
                batch.append(dataset[ix])
                batch_max = size
            else:
                self.clusters.append(batch)
                batch = [dataset[ix]]
        if len(batch) > 0: #最后一个mini batch可能还剩下一点东西
            self.clusters.append(batch)
        if reverse:
            self.clusters.reverse() #不同minibatch 组成的 token，但最大只能有6000个氨基酸

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, idx):
        # np.random.shuffle(self.clusters)
        return self.clusters[idx]


def batch_collate_function0(cluster):
    """
    A customized wrap up collate function
    Args:
        batch: a list of structure objects
    Shape:
        Output:
            coord_batch [B, 5, 4, 3] dtype=float32
            seq_batch   [B, L]       dtype=int64
            bert_mask_fraction_batch [B,] dtype=float32
            bert_mask_batch          [B, L]  dtype=torch.bool   0 represents mask, 1 no mask
            padding_mask_batch       [B, L]  dtype=torch.float32   0 represents mask, 1 no mask
    """
    batch = []
    for cluster_i in cluster:
        for i in cluster_i:
            batch.append(i)
    coord_batch = utils.CoordBatchConverter.collate_dense_tensors([i['coords'] for i in batch],0.0)
    seq_batch = utils.CoordBatchConverter.collate_dense_tensors([i['seq'] for i in batch],-1)
    cctop_batch = utils.CoordBatchConverter.collate_dense_tensors([i['cctop'] for i in batch],0)
    mask_seq_batch = utils.CoordBatchConverter.collate_dense_tensors([i['mask_seq'] for i in batch],0)
    padding_mask_batch = seq_batch!=-1 # True not mask, False represents mask
    seq_batch[~padding_mask_batch] = 0 # padding to 0
    padding_mask_batch = padding_mask_batch.to(torch.float32)
    length_batch = utils.CoordBatchConverter.collate_dense_tensors([i['length'] for i in batch],0)
    output = {
        "coord":coord_batch,
        "seq":seq_batch,
        "mask_seq":mask_seq_batch,
        "mask":padding_mask_batch,
        "cctop":cctop_batch,
        "length":length_batch
    }
    return output

def batch_collate_function(cluster):
    """
    A customized wrap up collate function
    Args:
        batch: a list of structure objects
    Shape:
        Output:
            coord_batch [B, L, n_atom, 3] dtype=float32
            seq_batch   [B, L]       dtype=int64
            bert_mask_fraction_batch [B,] dtype=float32
            bert_mask_batch          [B, L]  dtype=torch.bool   0 represents mask, 1 no mask
            padding_mask_batch       [B, L]  dtype=torch.float32   0 represents mask, 1 no mask
    """
    batch = []
    for cluster_i in cluster:
        for i in cluster_i:
            batch.append(i)
    coord_batch = utils.CoordBatchConverter.collate_dense_tensors([i['coords'] for i in batch],0.0)
    # #
    # coord_batch[:,:,[-2,-1],:] = coord_batch[:,:,[-1,-2],:]
    # NCANCANCA...
    bb_pos = torch.flatten(coord_batch[..., :3, :].permute(0,2,1,3), start_dim=-3, end_dim=-2)
    cctop_batch = utils.CoordBatchConverter.collate_dense_tensors([i['cctop'] for i in batch],0)
    seq_batch = utils.CoordBatchConverter.collate_dense_tensors([i['seq'] for i in batch],0)
    N_pos = utils.CoordBatchConverter.collate_dense_tensors([i['N_pos'] for i in batch],0.0)
    C_pos = utils.CoordBatchConverter.collate_dense_tensors([i['C_pos'] for i in batch],0.0)
    CA_pos = utils.CoordBatchConverter.collate_dense_tensors([i['CA_pos'] for i in batch],0.0)
    # frames = Rigid.from_3_points(C_pos, CA_pos, N_pos)
    mask = utils.CoordBatchConverter.collate_dense_tensors([i['mask'] for i in batch],0)
    residx = utils.CoordBatchConverter.collate_dense_tensors([i['residx'] for i in batch],0)
    length_batch = torch.stack([i['length'] for i in batch])
    output = {
        "seq":seq_batch,
        "length":length_batch,
        "coords":coord_batch,
        "N_pos":N_pos,
        "C_pos":C_pos,
        "CA_pos":CA_pos,
        "bb_pos":bb_pos,
        # "frames":frames,
        "mask":mask,
        "residx":residx,
        "cctop":cctop_batch,
    }
    return output

def batch_collate_function_nocluster(batch):
    """
    A customized wrap up collate function
    Args:
        batch: a list of structure objects
    Shape:
        Output:
            coord_batch [B, L, n_atom, 3] dtype=float32
            seq_batch   [B, L]       dtype=int64
            bert_mask_fraction_batch [B,] dtype=float32
            bert_mask_batch          [B, L]  dtype=torch.bool   0 represents mask, 1 no mask
            padding_mask_batch       [B, L]  dtype=torch.float32   0 represents mask, 1 no mask
    """
    # batch = []
    # for cluster_i in cluster:
    #     for i in cluster_i:
    #         batch.append(i)
    coord_batch = utils.CoordBatchConverter.collate_dense_tensors([i['coords'] for i in batch],0.0)
    # #
    # coord_batch[:,:,[-2,-1],:] = coord_batch[:,:,[-1,-2],:]
    # NCANCANCA...
    # B,L,3,3 - > torch.reshape (...,-1,3)
    bb_pos = torch.flatten(coord_batch[..., :3, :].permute(0,2,1,3), start_dim=-3, end_dim=-2)
    cctop_batch = utils.CoordBatchConverter.collate_dense_tensors([i['cctop'] for i in batch],0)
    seq_batch = utils.CoordBatchConverter.collate_dense_tensors([i['seq'] for i in batch],0)
    N_pos = utils.CoordBatchConverter.collate_dense_tensors([i['N_pos'] for i in batch],0.0)
    C_pos = utils.CoordBatchConverter.collate_dense_tensors([i['C_pos'] for i in batch],0.0)
    CA_pos = utils.CoordBatchConverter.collate_dense_tensors([i['CA_pos'] for i in batch],0.0)
    CB_pos = utils.CoordBatchConverter.collate_dense_tensors([i['CB_pos'] for i in batch],0.0)

    # frames = Rigid.from_3_points(C_pos, CA_pos, N_pos)
    mask = utils.CoordBatchConverter.collate_dense_tensors([i['mask'] for i in batch],0)
    residx = utils.CoordBatchConverter.collate_dense_tensors([i['residx'] for i in batch],0)
    length_batch = torch.stack([i['length'] for i in batch])
    name = [i['name'] for i in batch]
    output = {
        "seq":seq_batch,
        "length":length_batch,
        "coords":coord_batch,
        "N_pos":N_pos,
        "C_pos":C_pos,
        "CA_pos":CA_pos,
        "CB_pos":CB_pos,
        "bb_pos":bb_pos,
        # "frames":frames,
        "mask":mask,
        "residx":residx,
        "cctop":cctop_batch,
        "name":name,
    }
    return output

def collate_fn(batch):
    seqs, coords, lengths = [],[],[]
    for batch_i in batch:
        seqs.append(batch_i[0])
        lengths.append(batch_i[1])
        coords.append(batch_i[2])
    aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(sequences=seqs)
    target_positions, N_pos_tensor, CA_pos_tensor, C_pos_tensor = generate_label_tensors(coords)
    
    return aatype, mask, _residx, target_positions, N_pos_tensor, CA_pos_tensor, C_pos_tensor 

def generate_label_tensors(structures):
    all_pos_list = []
    N_pos_list, CA_pos_list, C_pos_list = [],[],[]
    for pos_dict in structures:
        # cat的顺序不重要，但一定要保证pred和targ一致，才能比较距离
        one_pos_tensor = torch.tensor(())
        for pos_i in zip(pos_dict['N'], pos_dict['CA'], pos_dict['C']):
            one_pos_tensor = torch.cat((one_pos_tensor, torch.tensor(pos_i)), dim=0)
        all_pos_list.append(one_pos_tensor)
        N_pos_list.append(torch.tensor(pos_dict['N']))
        CA_pos_list.append(torch.tensor(pos_dict['CA']))
        C_pos_list.append(torch.tensor(pos_dict['C']))

    N_pos_tensor = collate_dense_tensors(N_pos_list)
    CA_pos_tensor = collate_dense_tensors(CA_pos_list)
    C_pos_tensor = collate_dense_tensors(C_pos_list)
    target_positions = collate_dense_tensors(all_pos_list)

    return target_positions, N_pos_tensor, CA_pos_tensor, C_pos_tensor

def batch_collate_function_withname(cluster):
    """
    A customized wrap up collate function
    Args:
        batch: a list of structure objects
    Shape:
        Output:
            coord_batch [B, L, n_atom, 3] dtype=float32
            seq_batch   [B, L]       dtype=int64
            bert_mask_fraction_batch [B,] dtype=float32
            bert_mask_batch          [B, L]  dtype=torch.bool   0 represents mask, 1 no mask
            padding_mask_batch       [B, L]  dtype=torch.float32   0 represents mask, 1 no mask
    """
    batch = []
    for cluster_i in cluster:
        for i in cluster_i:
            batch.append(i)
    coord_batch = utils.CoordBatchConverter.collate_dense_tensors([i['coords'] for i in batch],0.0)
    # #
    # coord_batch[:,:,[-2,-1],:] = coord_batch[:,:,[-1,-2],:]
    # NCANCANCA...
    bb_pos = torch.flatten(coord_batch[..., :3, :].permute(0,2,1,3), start_dim=-3, end_dim=-2)
    cctop_batch = utils.CoordBatchConverter.collate_dense_tensors([i['cctop'] for i in batch],0)
    seq_batch = utils.CoordBatchConverter.collate_dense_tensors([i['seq'] for i in batch],0)
    N_pos = utils.CoordBatchConverter.collate_dense_tensors([i['N_pos'] for i in batch],0.0)
    C_pos = utils.CoordBatchConverter.collate_dense_tensors([i['C_pos'] for i in batch],0.0)
    CA_pos = utils.CoordBatchConverter.collate_dense_tensors([i['CA_pos'] for i in batch],0.0)
    # frames = Rigid.from_3_points(C_pos, CA_pos, N_pos)
    mask = utils.CoordBatchConverter.collate_dense_tensors([i['mask'] for i in batch],0)
    residx = utils.CoordBatchConverter.collate_dense_tensors([i['residx'] for i in batch],0)
    length_batch = torch.stack([i['length'] for i in batch])
    name = [i['name'] for i in batch]
    output = {
        "seq":seq_batch,
        "length":length_batch,
        "coords":coord_batch,
        "N_pos":N_pos,
        "C_pos":C_pos,
        "CA_pos":CA_pos,
        "bb_pos":bb_pos,
        # "frames":frames,
        "mask":mask,
        "residx":residx,
        "cctop":cctop_batch,
        "name":name,
    }
    return output

def load_name_seq_CA(jsonl_file):
        sequence, CA_pososition = [],[]
        dset = utils.load_jsonl(jsonl_file)        
        for entry in dset:
            name = entry['name']
            seq = entry['seq']
            CA_pos = torch.tensor(entry['coords']['CA'])
            sequence.append((name,seq))
            CA_pososition.append(CA_pos)

        return sequence, CA_pososition

def load_name_seq_C(jsonl_file, max_length):
        sequence,N_position, CA_position, C_position = [],[],[],[]
        dset = utils.load_jsonl(jsonl_file)
        for entry in dset:
            name = entry['name']
            seq = entry['seq']
            length = torch.tensor([len(seq)],dtype=torch.long)
            bad_chars = set([s for s in seq]).difference(utils.restypes)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    pass
                else:
                    continue
            else:
                continue        
            if entry['coords'].shape[0] != len(seq):
                print('not_match')
                continue
            CA_pos = torch.tensor(entry['coords']['CA'])
            N_pos = torch.tensor(entry['coords']['N'])
            C_pos = torch.tensor(entry['coords']['C'])
            sequence.append((name,seq))
            CA_position.append(CA_pos)
            N_position.append(N_pos)
            C_position.append(C_pos)
        return sequence, N_position, CA_position, C_position


class Crop_StructureDataset(Dataset):
    def __init__(self,jsonl_file,max_length=500, crop_size=None):
        self.crop_size = crop_size
        dset = utils.load_jsonl(jsonl_file)
        for i in dset:
            if i['name'].startswith("AF"):
                i['name'] += "_A"
            i['cctop'] = i['cctop'].replace("T","S") #替换减少一类
        cctop_code = 'IMOULS'
        self.data = []
        self.discard = {"bad_chars":0,"too_long":0, "not_match":0}
        
        for entry in dset:
            name = entry['name']
            seq = entry['seq']
            cctop = entry['cctop']
            length = torch.tensor([len(seq)],dtype=torch.long)
            # Check  if in alphabet
            bad_chars = set([s for s in seq]).difference(utils.restypes)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    pass
                else:
                    self.discard['too_long'] += 1
                    continue
            else:
                # print(entry['name'], bad_chars, entry['seq'])
                self.discard['bad_chars'] += 1
                continue
            seq = torch.tensor([restype_order[i] for i in seq],dtype=torch.long)
            cctop = torch.tensor([cctop_code.index(i) for i in cctop],dtype=torch.long)
            coords = torch.from_numpy(np.stack(list(entry['coords'].values()),axis=-2))
            coords = coords.to(torch.float32)
            coords = utils.nan_to_num(coords) # remove the nan value
            N_pos = torch.tensor(entry['coords']['N'])
            C_pos = torch.tensor(entry['coords']['C'])
            CA_pos = torch.tensor(entry['coords']['CA'])
            CB_pos = torch.tensor(entry['coords']['CB'])            
            mask = torch.ones(length.item(), dtype=int)
            residx = torch.arange(length.item(), dtype=int)
            if coords.shape[0] != seq.shape[0] or cctop.shape[0] != seq.shape[0]:
                self.discard['not_match'] += 1
                continue
            self.data.append({
                "name":name,
                "coords":coords,
                "N_pos":N_pos,
                "C_pos":C_pos,
                "CA_pos":CA_pos,
                "CB_pos": CB_pos,
                "seq":seq,
                "cctop":cctop,
                "mask":mask,
                "residx":residx,
                "length":length
            })
        print(f"UNK token:{self.discard['bad_chars']},too long:{self.discard['too_long']}, 'not_match':{self.discard['not_match']}")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        if self.crop_size is None:
            return self.data[idx]
        else:
            # 4.03 by zb : seems to have some memory efficiency problem?
            crop_idx = get_single_crop_idx(
                num_res=self.data[idx]["length"], crop_size=self.crop_size)
            crop_data = {}
            for key in self.data[idx]:
                if not isinstance(self.data[idx][key], torch.Tensor) or key == 'length':
                    crop_data[key] = self.data[idx][key]
                else:
                    crop_data[key] = self.data[idx][key][crop_idx]
            return crop_data
