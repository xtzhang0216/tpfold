from pathlib import Path
import os
import torch
import esm
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import json
import argparse
from esm.modules import apc, symmetrize
from data import load_name_seq_C
import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist

# 找出一维ndarray里最大值的索引

def find_max_index(tensor):
    max_value = torch.max(tensor)
    max_index = torch.nonzero(tensor == max_value)
    return max_index

parser = argparse.ArgumentParser()
# parser.add_argument('--write_path',type=str,default="/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/15B_plddt_design/", help="name")
# parser.add_argument('--read_path',type=str,default="/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/original_design/", help="learning rate of Adam optimizer")
# parser.add_argument('--pdb_path',type=str,default="/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/predict/original_design/", help="learning rate of Adam optimizer")
parser.add_argument('--model',type=str,default="mlmp30", help="learning rate of Adam optimizer")
parser.add_argument('--parameters',type=str,default=None, help="learning rate of Adam optimizer")

args = parser.parse_args()

# esmdata = torch.load("/lustre/gst/xuchunfu/zhangxt/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D-contact-regression.pt")
# tpmdata = torch.load("/lustre/gst/xuchunfu/zhangxt/myesm/regression_esm.pt")

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

def compute_contact_CA(CA_pos_list,
    cutoff=8.0,
    eps=1e-6,
    **kwargs,):
    all_contact_map = torch.tensor([])
    for CA_pos in CA_pos_list:
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
        contact_map = contact_map.reshape(-1, 1)
        all_contact_map = torch.cat((all_contact_map, contact_map), dim=0)
    return all_contact_map
num_layers, num_heads = 36, 40
seqs, N_poses, CA_poses, C_poses = load_name_seq_C("/lustre/gst/xuchunfu/zhangxt/data/native20.jsonl")
class ContactPredictionHead(nn.Module):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(
        self,
        in_features: int,
        prepend_bos: bool,
        append_eos: bool,
        bias=True,
        eos_idx: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        if append_eos and eos_idx is None:
            raise ValueError("Using an alphabet with eos token, but no eos token was passed in.")
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, 1, bias)
        self.activation = nn.Sigmoid()

    def forward(self, tokens, attentions):
        # remove eos token attentions
        if self.append_eos:
            eos_mask = tokens.ne(self.eos_idx).to(attentions)
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            attentions = attentions * eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        if self.prepend_bos:
            attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # features: B x C x T x T
        attentions = attentions.to(
            self.regression.weight.device
        )  # attentions always float32, may need to convert to float16
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        return attentions

tmContactPredictionHead = ContactPredictionHead(
        in_features=36*40,
        prepend_bos=True,
        append_eos=True,
        eos_idx=2,
    )


def load_plm(esm2_path, ft_path=None):
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
 
        regression_data = load_regression_hub('esm2_t36_3B_UR50D')
        esm_model, esm_dict = load_model_and_alphabet_core('esm2_t36_3B_UR50D', model_data, regression_data)
        return esm_model, esm_dict
esm_model, esm_dict = load_plm(
        esm2_path="/lustre/gst/xuchunfu/zhangxt/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt",
        ft_path=args.parameters,
    )
# esm_model, esm_dict = esm.pretrained.esm2_t36_3B_UR50D()
esm_model = esm_model.eval().cuda()

# model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = esm_dict.get_batch_converter()
# batch_lens = (batch_tokens != esm_dict.padding_idx).sum(1)
attention_maps = torch.tensor([])
contact_maps = torch.tensor([])
for seq, N_pos, CA_pos, C_pos in zip(seqs, N_poses, CA_poses, C_poses):
    batch_labels, batch_strs, batch_tokens = batch_converter([seq])
    batch_tokens = batch_tokens.cuda()
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[36], need_head_weights=True)
    attention_map = tmContactPredictionHead(batch_tokens, results["attentions"])
    attention_map = attention_map.reshape(-1, num_layers*num_heads)
    attention_maps = torch.cat((attention_maps, attention_map), dim=0)
    contact_map = torch.tensor(compute_contact_CB(N_pos, CA_pos, C_pos)).reshape(-1, 1)
    contact_maps = torch.cat((contact_maps, contact_map), dim=0)
# # 2D feature array, each entry is the attentions for one contact (i, j) for one protein
# X = [N x (num_layers * num_attn_heads)]
# #  2D boolean array, whether or not each entry corresponds to a contact
# y = [N x 1]

clf = LogisticRegression(
    penalty="l1",
    C=0.15,
    solver="liblinear",
)

# contact_map = compute_contact_CB(CA_poses)

clf.fit(attention_maps, contact_maps)

torch.save({"model":
            {"contact_head.regression.weight":torch.tensor(clf.coef_), "contact_head.regression.bias":torch.tensor(clf.intercept_)}
            },
            f"/lustre/gst/xuchunfu/zhangxt/checkpoint/regression_afdb/regression_{args.model}.pt")
# def read_pseudo_contact_map(infile="model1.pdb",atom_sele="CB",
#     cutoff=8, sep_range=str(short_range_def),offset=0):
#     res_dist_list=calc_res_dist(infile,atom_sele)
#     res_con_list=calc_res_contact(res_dist_list,sep_range,cutoff)
#     if len(res_con_list)==0:
#         return zip([],[],[])
#     resi1,resi2,p=map(list,zip(*res_con_list))
#     for i in range(len(res_con_list)):
#         resi1[i]+=offset
#         resi2[i]+=offset
#         p[i]=1-1.*p[i]/cutoff
#     p,resi1,resi2=map(list,zip(*sorted(zip(p,resi1,resi2),reverse=True)))
#     return zip(resi1,resi2,p)

