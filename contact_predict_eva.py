# partitioned_trues = [partition_contacts(contact_map) for contact_map in true_maps]
# partitioned_preds = [partition_contacts(contact_probs) for contact_probs in predicted_probs]

# _long_preds = list(zip(*partitioned_preds))[2]
# _long_trues = list(zip(*partitioned_trues))[2]

# # Filter out sequences whose length is less than 24
# # They show up as empty lists in list of long contacts/predictions
# long_preds = []
# for l in _long_preds:
#     if len(l) > 0:
#         long_preds.append(l)
        
# long_trues = []
# for l in _long_trues:
#     if len(l) > 0:
#         long_trues.append(l)

# precision, recall, f1, aupr, precision_L, precision_L_2, precision_L_5 = collect_metrics(long_trues, long_preds)

# # Report 
# long_results.loc[model, 'precision'] = np.mean(precision)
# long_results.loc[model, 'recall'] = np.mean(recall)
# long_results.loc[model, 'f1'] = np.mean(f1)
# long_results.loc[model, 'aupr'] = np.mean(aupr)
# long_results.loc[model, 'precision@L'] = np.mean(precision_L)
# long_results.loc[model, 'precision@L/2'] = np.mean(precision_L_2)
# long_results.loc[model, 'precision@L/5'] = np.mean(precision_L_5)

# # Turn into pd.DataFrame and dump to csv 
# short_results.round(2).to_csv('tables/short_range_contact_results.csv') 
# medium_results.round(2).to_csv('tables/medium_range_contact_results.csv') 
# long_results.round(2).to_csv('tables/long_range_contact_results.csv') 

from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import itertools
import os
import string
from pathlib import Path
import argparse

import numpy as np
import torch
from scipy.spatial.distance import squareform, pdist, cdist
import matplotlib.pyplot as plt
import matplotlib as mpl
from Bio import SeqIO
import biotite.structure as bs
from biotite.structure.io.pdbx import PDBxFile, get_structure
from biotite.database import rcsb
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import esm
from data import load_name_seq_C

parser = argparse.ArgumentParser()
# parser.add_argument('--write_path',type=str,default="/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/15B_plddt_design/", help="name")
# parser.add_argument('--read_path',type=str,default="/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/original_design/", help="learning rate of Adam optimizer")
# parser.add_argument('--pdb_path',type=str,default="/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/predict/original_design/", help="learning rate of Adam optimizer")
parser.add_argument('--model',type=str,default="/lustre/gst/xuchunfu/zhangxt/.cache/huggingface/hub/models--facebook--esm2_t48_15B_UR50D/snapshots/5fbca39631164edc1d402a5aa369f982f72ee282/", help="learning rate of Adam optimizer")
parser.add_argument('--ft_path',type=str,default=None, help="learning rate of Adam optimizer")
parser.add_argument('--regression_path',type=str,default="/lustre/gst/xuchunfu/zhangxt/checkpoint/regression/regression_mlmp60.pt", help="learning rate of Adam optimizer")
args = parser.parse_args()
def extend(a, b, c, L, A, D):
    """
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    """

    def normalize(x):
        return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return c + sum([m * d for m, d in zip(m, d)])
def compute_contact_CB(
    N, CA, C,
    distance_threshold: float = 8.0,
    chain: Optional[str] = None,
) -> np.ndarray:


    Cbeta = extend(C, N, CA, 1.522, 1.927, -2.143)
    dist = squareform(pdist(Cbeta))
    
    contacts = dist < distance_threshold
    contacts = contacts.astype(np.int64)
    contacts[np.isnan(dist)] = -1
    return contacts
# esmdata = torch.load("/lustre/gst/xuchunfu/zhangxt/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D-contact-regression.pt")
# tpmdata = torch.load("/lustre/gst/xuchunfu/zhangxt/checkpoint/regression/regression_mlmp60.pt")
def compute_contact_CA(CA_pos,
    cutoff=8.0,
    eps=1e-6,
    **kwargs,):
    distance_map = torch.sqrt(
        eps
        + torch.sum(
            (
                CA_pos[..., None, :]
                - CA_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    contact_map = (distance_map < cutoff).float()
        
    return contact_map

def compute_precisions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: Optional[torch.Tensor] = None,
    minsep: int = 6,
    maxsep: Optional[int] = None,
    override_length: Optional[int] = None,  # for casp
):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)
    override_length = (targets[0, 0] >= 0).sum()

    # Check sizes
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, "
            f"targets of size {targets.size()}"
        )
    device = predictions.device

    batch_size, seqlen, _ = predictions.size()
    seqlen_range = torch.arange(seqlen, device=device)

    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    valid_mask = sep >= minsep
    valid_mask = valid_mask & (targets >= 0)  # negative targets are invalid

    if maxsep is not None:
        valid_mask &= sep < maxsep

    if src_lengths is not None:
        valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
        valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
    else:
        src_lengths = torch.full([batch_size], seqlen, device=device, dtype=torch.long)

    predictions = predictions.masked_fill(~valid_mask, float("-inf"))

    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    topk = seqlen if override_length is None else max(seqlen, override_length)
    indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
    topk_targets = targets_upper[torch.arange(batch_size).unsqueeze(1), indices]
    if topk_targets.size(1) < topk:
        topk_targets = F.pad(topk_targets, [0, topk - topk_targets.size(1)])

    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

    gather_lengths = src_lengths.unsqueeze(1)
    if override_length is not None:
        gather_lengths = override_length * torch.ones_like(
            gather_lengths, device=device
        )

    gather_indices = (
        torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths
    ).type(torch.long) - 1

    binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(
        binned_cumulative_dist
    )

    pl5 = binned_precisions[:, 1]
    pl2 = binned_precisions[:, 4]
    pl = binned_precisions[:, 9]
    auc = binned_precisions.mean(-1)

    return {"AUC": auc, "P@L": pl, "P@L2": pl2, "P@L5": pl5}

def evaluate_prediction(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    contact_ranges = [
        ("local", 3, 6),
        ("short", 6, 12),
        ("medium", 12, 24),
        ("long", 24, None),
    ]
    metrics = {}
    targets = targets.to(predictions.device)
    for name, minsep, maxsep in contact_ranges:
        rangemetrics = compute_precisions(
            predictions,
            targets,
            minsep=minsep,
            maxsep=maxsep,
        )
        for key, val in rangemetrics.items():
            metrics[f"{name}_{key}"] = val.item()
    return metrics

def load_plm(esm2_path,  regression_path, ft_path=None):
        from esm.pretrained import load_regression_hub, load_model_and_alphabet_core
        model_data = torch.load(esm2_path)
        if ft_path is not None:
            ft_model_state = torch.load(ft_path)
            layer35_state = {
                'encoder.sentence_encoder.layers.35.self_attn.k_proj.weight': ft_model_state['esm.encoder.layer.35.attention.self.key.weight'],
                'encoder.sentence_encoder.layers.35.self_attn.k_proj.bias': ft_model_state['esm.encoder.layer.35.attention.self.key.bias'],
                'encoder.sentence_encoder.layers.35.self_attn.v_proj.weight': ft_model_state['esm.encoder.layer.35.attention.self.value.weight'],
                'encoder.sentence_encoder.layers.35.self_attn.v_proj.bias': ft_model_state['esm.encoder.layer.35.attention.self.value.bias'],
                'encoder.sentence_encoder.layers.35.self_attn.q_proj.weight': ft_model_state['esm.encoder.layer.35.attention.self.query.weight'],
                'encoder.sentence_encoder.layers.35.self_attn.q_proj.bias': ft_model_state['esm.encoder.layer.35.attention.self.query.bias'],
                'encoder.sentence_encoder.layers.35.self_attn.out_proj.weight':ft_model_state['esm.encoder.layer.35.attention.output.dense.weight'],
                'encoder.sentence_encoder.layers.35.self_attn.out_proj.bias':ft_model_state['esm.encoder.layer.35.attention.output.dense.bias'],
                'encoder.sentence_encoder.layers.35.self_attn.rot_emb.inv_freq':ft_model_state['esm.encoder.layer.35.attention.self.rotary_embeddings.inv_freq'],
                'encoder.sentence_encoder.layers.35.self_attn_layer_norm.weight':ft_model_state['esm.encoder.layer.35.attention.LayerNorm.weight'],
                'encoder.sentence_encoder.layers.35.self_attn_layer_norm.bias':ft_model_state['esm.encoder.layer.35.attention.LayerNorm.bias'],
                'encoder.sentence_encoder.layers.35.fc1.weight':ft_model_state['esm.encoder.layer.35.intermediate.dense.weight'],
                'encoder.sentence_encoder.layers.35.fc1.bias':ft_model_state['esm.encoder.layer.35.intermediate.dense.bias'],
                'encoder.sentence_encoder.layers.35.fc2.weight':ft_model_state['esm.encoder.layer.35.output.dense.weight'],
                'encoder.sentence_encoder.layers.35.fc2.bias':ft_model_state['esm.encoder.layer.35.output.dense.bias'],
                'encoder.sentence_encoder.layers.35.final_layer_norm.weight':ft_model_state['esm.encoder.layer.35.LayerNorm.weight'],
                'encoder.sentence_encoder.layers.35.final_layer_norm.bias':ft_model_state['esm.encoder.layer.35.LayerNorm.bias'],
                }
            # 'esm.encoder.emb_layer_norm_after.weight', 'esm.encoder.emb_layer_norm_after.bias'

            model_data['model'].update(layer35_state)
        regression_data = torch.load(regression_path)
        esm_model, esm_dict = load_model_and_alphabet_core('esm2_t36_3B_UR50D', model_data, regression_data)
        return esm_model, esm_dict

esm_model, esm_dict = load_plm(
        esm2_path="/lustre/gst/xuchunfu/zhangxt/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt",
        ft_path=args.ft_path,
        regression_path="/lustre/gst/xuchunfu/zhangxt/checkpoint/regression/regression_"+args.model+'.pt',
    )
# esm_model, esm_dict = esm.pretrained.esm2_t36_3B_UR50D()
esm_model = esm_model.eval().cuda()

def compute_precisions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: Optional[torch.Tensor] = None,
    minsep: int = 6,
    maxsep: Optional[int] = None,
    override_length: Optional[int] = None,  # for casp
):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)
    override_length = (targets[0, 0] >= 0).sum()

    # Check sizes
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, "
            f"targets of size {targets.size()}"
        )
    device = predictions.device

    batch_size, seqlen, _ = predictions.size()
    seqlen_range = torch.arange(seqlen, device=device)

    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    valid_mask = sep >= minsep
    valid_mask = valid_mask & (targets >= 0)  # negative targets are invalid

    if maxsep is not None:
        valid_mask &= sep < maxsep

    if src_lengths is not None:
        valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
        valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
    else:
        src_lengths = torch.full([batch_size], seqlen, device=device, dtype=torch.long)

    predictions = predictions.masked_fill(~valid_mask, float("-inf"))

    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    topk = seqlen if override_length is None else max(seqlen, override_length)
    indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
    topk_targets = targets_upper[torch.arange(batch_size).unsqueeze(1), indices]
    if topk_targets.size(1) < topk:
        topk_targets = F.pad(topk_targets, [0, topk - topk_targets.size(1)])

    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

    gather_lengths = src_lengths.unsqueeze(1)
    if override_length is not None:
        gather_lengths = override_length * torch.ones_like(
            gather_lengths, device=device
        )

    gather_indices = (
        torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths
    ).type(torch.long) - 1

    binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(
        binned_cumulative_dist
    )

    pl5 = binned_precisions[:, 1]
    pl2 = binned_precisions[:, 4]
    pl = binned_precisions[:, 9]
    auc = binned_precisions.mean(-1)

    return {"AUC": auc, "P@L": pl, "P@L2": pl2, "P@L5": pl5}


def evaluate_prediction(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    contact_ranges = [
        ("local", 3, 6),
        ("short", 6, 12),
        ("medium", 12, 24),
        ("long", 24, None),
    ]
    metrics = {}
    targets = targets.to(predictions.device)
    for name, minsep, maxsep in contact_ranges:
        rangemetrics = compute_precisions(
            predictions,
            targets,
            minsep=minsep,
            maxsep=maxsep,
        )
        for key, val in rangemetrics.items():
            metrics[f"{name}_{key}"] = val.item()
    return metrics
seqs, N_poses, CA_poses, C_poses  = load_name_seq_C("/lustre/gst/xuchunfu/zhangxt/data/native20.jsonl")
batch_converter = esm_dict.get_batch_converter()
result_sum = {'local_AUC': 0, 'local_P@L': 0, 'local_P@L2': 0, 'local_P@L5': 0, 'short_AUC': 0, 'short_P@L': 0, 'short_P@L2': 0, 'short_P@L5': 0, 'medium_AUC': 0, 'medium_P@L': 0, 'medium_P@L2': 0, 'medium_P@L5': 0, 'long_AUC': 0, 'long_P@L': 0, 'long_P@L2': 0, 'long_P@L5': 0}
for seq, N, CA, C in zip(seqs,N_poses, CA_poses, C_poses):
    batch_labels, batch_strs, batch_tokens = batch_converter([seq])
    batch_tokens = batch_tokens.cuda()
    with torch.no_grad():
        predict_map = esm_model.predict_contacts(batch_tokens)[0]
    contact_map = compute_contact_CB(N, CA, C)
    result = evaluate_prediction(predict_map, contact_map)
    for key in result_sum.keys():
        result_sum[key] += result[key]
# 将字典中的值除以样本数
for key in result_sum.keys():
    result_sum[key] /= len(seqs)
 
esm2_results = pd.DataFrame(result_sum, index=args.model)

# 将result_sum写入csv文件末行，该结果索引为mlmp
esm2_results.to_csv('/lustre/gst/xuchunfu/zhangxt/result.csv', mode='a', header=False, index=False)
