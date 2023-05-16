import numpy as np
import torch
import json
tm={'n':torch.rand(5,3),'c':torch.rand(5,3),'c1':torch.rand(5,3)}
from openfold.utils.rigid_utils import Rigid
# with open('/data/home/scv6707/run/zxt/data/TMfromzb/json/content.jsonl') as f:
#     for items in jsonlines.Reader(f):
#         protdict = items
#         break
# coordinates = protdict['coords']
# turerigid = Rigid.from_3_points(torch.tensor(coordinates['N']), torch.tensor(coordinates['CA']), torch.tensor(coordinates['C']))
import torch
from openfold.utils.rigid_utils import (
    Rotation,
    Rigid,
)
from openfold.utils.loss import (
    torsion_angle_loss,
    compute_fape)
# 2个蛋白(batch)，一个长为4，一个6
# frames有6个(maxlen),atoms有3*6=18
batch_size = 4
n_frames = 500
n_atoms = 1500
tm={'length':torch.tensor([4,6]), 'N':torch.rand(2,6,3),'CA':torch.rand(2,6,3),'C':torch.rand(2,6,3)}
# t_gt = Rigid.from_3_points(tm['N'], tm['CA'], tm['C'])
x_gt = torch.cat((tm['N'], tm['CA'], tm['C']), dim=1)
x = torch.rand((batch_size, n_atoms, 3))
# x_gt = torch.rand((batch_size, n_atoms, 3))
rots = torch.rand((batch_size, n_frames, 3, 3))
# rots_gt = torch.rand((batch_size, n_frames, 3, 3))
trans = torch.rand((batch_size, n_frames, 3))
# trans_gt = torch.rand((batch_size, n_frames, 3))
t = Rigid(Rotation(rot_mats=rots), trans)
# t_gt = Rigid(Rotation(rot_mats=rots_gt), trans_gt)
frames_mask = torch.randint(0, 2, (batch_size, n_frames)).float()
positions_mask = torch.randint(0, 2, (batch_size, n_atoms)).float()
length_scale = 10

# loss = compute_fape(
#     pred_frames=t,
#     target_frames=t_gt,
#     frames_mask=frames_mask,
#     pred_positions=x,
#     target_positions=x_gt,
#     positions_mask=positions_mask,
#     length_scale=length_scale,
# )

def create_batched_sequence_datasest(
    file_path, max_tokens_per_batch: int = 1024 #控制批训练大小
):
    with open(file_path, "r") as f:
        examples = [json.loads(line) for line in f.readlines()]

    name, seq, coords, length = [], [], [], []
    for example in examples:
        name.append(example['name'])
        seq.append(example['seq'])
        coords.append(example['coords'])
        length.append(example['length'])

    batch_headers, batch_sequences, batch_structures, num_tokens = [], [], [], 0
    for header, sequence, structure in zip(name, seq, coords):
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences, batch_structures
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(sequence)
        batch_structures.append(structure)
        num_tokens += len(seq)

    yield batch_headers, batch_sequences, batch_structures
def collate_dense_tensors(
    samples, pad_v: float = 0
) -> torch.Tensor:
    """
    Takes a list of tensors with the following dimensions:
        [(d_11,       ...,           d_1K),
         (d_21,       ...,           d_2K),
         ...,
         (d_N1,       ...,           d_NK)]
    and stack + pads them into a single tensor of:
    (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
    """
    if len(samples) == 0:
        return torch.Tensor()
    if len(set(x.dim() for x in samples)) != 1:
        raise RuntimeError(
            f"Samples has varying dimensions: {[x.dim() for x in samples]}"
        )
    (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
    max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
    result = torch.empty(
        len(samples), *max_shape, dtype=samples[0].dtype, device=device
    ) #[b,ml]
    result.fill_(pad_v)
    for i in range(len(samples)):
        result_i = result[i]
        t = samples[i]
        result_i[tuple(slice(0, k) for k in t.shape)] = t
    return result

def generate_label(structures):
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
    target_frames =  Rigid.from_3_points(N_pos_tensor, CA_pos_tensor, C_pos_tensor)

    return target_positions, target_frames
# s = torch.rand(2,23,7)
# rigids = Rigid.from_tensor_7(
#             s
#         )
# backb_to_global = Rigid(
#                 Rotation(
#                     rot_mats=rigids.get_rots().get_rot_mats(), 
#                     quats=None
#                 ),
#                 rigids.get_trans(),
#             )
# batch_proteins = create_batched_sequence_datasest('/data/home/scv6707/run/zxt/data/TMfromzb/json/content.jsonl')
# for headers, sequences, structures in batch_proteins:
#     # headers2 = collate_dense_tensors(headers)
#     target_positions, target_frames = generate_label(structures)
#     loss = compute_fape(
#     pred_frames=t,
#     target_frames=target_frames,
#     frames_mask=frames_mask,
#     pred_positions=x,
#     target_positions=target_positions,
#     positions_mask=positions_mask,
#     length_scale=length_scale,
# )

N = torch.tensor([[[1,2,4]]])
CA = torch.tensor([[[4,7,6.0]]])
C = torch.tensor([[[5,5,9]]])

# N = torch.rand(1,1,3)
# CA = torch.rand(1,1,3)
# C = torch.rand(1,1,3)
# t_gt, rot1 = Rigid.from_3_points(C,CA,N)

from openfold.utils.rigid_utils import rot_vec_mul
input = torch.tensor([[[1.0,2.0,3.0]]])
rotation = torch.rand(1,1,3,3)
rot_vec_mul(rotation, input)