from symbol import atom
import os
import numpy as np
from Bio.PDB import *
import Bio.PDB
from typing import Union
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from typing import Sequence, Tuple, List, Optional,Iterable
from esm.data import BatchConverter


# A number of functions/classes are adopted from these two code pages:
# 1. https://github.com/jingraham/neurips19-graph-protein-design
# 2. https://github.com/dauparas/ProteinMPNN/blob/main/protein_mpnn_utils.py

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

def norm(tensor, dim, eps=1e-8, keepdim=False):
    """
    Returns L2 norm along a dimension.
    """
    return torch.sqrt(
        torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)


def normalize(tensor, dim=-1):
    """
    Normalizes a tensor along a dimension after removing nans.
    """
    return nan_to_num(
        torch.div(tensor, norm(tensor, dim=dim, keepdim=True))
    )


def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.    
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)

def structure_parser(protein: Union[str, Bio.PDB.Structure.Structure]) -> list:
    parser = PDBParser()
    if isinstance(protein, str):
        protein = parser.get_structure("parser", file=protein)

    coord = []
    for chain in protein.get_chains():
        if list(chain.get_residues())[0].id[0] != " ":  # drop other chains
            continue
        N = []
        CA = []
        C = []
        O = []
        for residue in chain.get_residues():
            if residue.has_id("O") and residue.has_id("CA") and residue.has_id("C") and residue.has_id("N") and alphabet(residue) != "X" and isinstance(residue, Bio.PDB.Residue.Residue):
                atoms = list(residue.get_atoms())
                N.append(list(atoms[0].get_vector()))
                CA.append(list(atoms[1].get_vector()))
                C.append(list(atoms[2].get_vector()))
                O.append(list(atoms[3].get_vector()))
        if chain.id == " ":
            coord.append({"None": (N, CA, C, O)})
        else:
            coord.append({chain.id: (N, CA, C, O)})
    return coord


def alphabet(res: Bio.PDB.Residue.Residue) -> str:
    if not isinstance(res, Bio.PDB.Residue.Residue):
        raise TypeError("Not correct input")
    code_standard = {
        'ALA': 'A', 'VAL': 'V', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
        'ILE': 'I', 'LEU': 'L', 'ASP': 'D', 'GLU': 'E', 'LYS': 'K',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'TYR': 'Y', 'HIS': 'H',
        'CYS': 'C', 'ASN': 'N', 'GLN': 'Q', 'TRP': 'W', 'GLY': 'G',
    }
    # standard aa & not hetero residue
    if res.get_resname() in code_standard.keys() and res.id[0] == " ":
        return code_standard[res.get_resname()]
    else:
        return "X"


def sequence_parse(structure: Union[str, Bio.PDB.Structure.Structure]) -> list:
    parser = PDBParser()
    if isinstance(structure, str):
        structure = parser.get_structure("parser", file=structure)

    sequence_list = []

    for chain in structure.get_chains():
        # exclude which contains water or hetero molecules
        if list(chain.get_residues())[0].id[0] != " ":
            continue
        seq = ""
        num = 0
        for res in chain.get_residues():
            if alphabet(res) != "X" and isinstance(res, Bio.PDB.Residue.Residue):
                seq += alphabet(res)
                num += 1
        if chain.id == " ":
            sequence_list.append({"None": (seq, num)})
        else:
            sequence_list.append({chain.id: (seq, num)})

    return sequence_list


def load_jsonl(json_file: str) -> list:
    data = []
    with open(json_file, "r") as f:
        for line in f:
            try:
                # 每一行代表一个序列的字典字符串加一个'\n',对于字符串应该用json.loads()文件进行读取
                data.append(json.loads(line.replace("\n", "")))
            except ValueError:
                pass
    return data


def write_jsonl(input_dir=None, output_dir=None, name_output=False):
    file_name = "load_pdb.jsonl"
    name_list = []
    dataset = []
    input_list = os.listdir(input_dir)
    for file in input_list:
        if ".pdb" in file:
            try:
                protein = os.path.join(input_dir, file)
                sequence_list = sequence_parse(protein)
                structure_list = structure_parser(protein)
                for i in range(len(sequence_list)):
                    # dict {chain_id : (seq,length)}
                    sequence = sequence_list[i]
                    chain_id = list(sequence.keys())[0]
                    seq, length = sequence[chain_id]

                    # dict {chain_id : (N,CA,C,O)}
                    structure = structure_list[i]
                    chain_backup = list(structure.keys())[0]
                    N, CA, C, O = structure[chain_backup]

                    if chain_backup != chain_id:  # check pdb files
                        raise ValueError(
                            f"{chain_id} not equals to {chain_backup},check {file}")

                    if chain_id != " ":
                        name = file.replace(".pdb", f"_{chain_id}")
                    else:
                        name = file.replace(".pdb", "")

                    entity = {
                        "seq": seq,
                        "coords": {
                            "N": N,
                            "CA": CA,
                            "C": C,
                            "O": O,
                        },
                        "name": name,
                        "length": length,
                        "chain": chain_id
                    }

                    dataset.append(entity)
                    name_list.append(name)
            except:
                pass

    name_dic = {"data": name_list}

    outfile1 = os.path.join(output_dir, file_name)
    with open(outfile1, 'w') as f1:
        for entry in dataset:
            f1.write(json.dumps(entry) + '\n')
    if name_output:
        outfile2 = os.path.join(output_dir, "protein_list.json")
        with open(outfile2, "w") as f2:
            f2.write(json.dumps(name_dic))


def featurize(batch, device, shuffle_fraction=0.,mask_fraction=1.0,mask_min=0.6):
    """ Pack and pad batch into torch tensors """
    # pdb file {"seq":AAA,"coords"{},"name":5cbo_A,"length":100,"chain":"A"}
    # alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    cctop_code = 'IMOSL'

    B = len(batch)
    lengths = np.array([b['length'] for b in batch], dtype=np.int32)
    L_max = max([b['length'] for b in batch])
    X = np.zeros([B, L_max, 5, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    C = np.zeros([B, L_max], dtype=np.int32)

    def shuffle_subset(n, p):
        n_shuffle = np.random.binomial(n, p)
        ix = np.arange(n)
        ix_subset = np.random.choice(ix, size=n_shuffle, replace=False)
        ix_subset_shuffled = np.copy(ix_subset)
        np.random.shuffle(ix_subset_shuffled)
        ix[ix_subset] = ix_subset_shuffled
        return ix

    # Build the batch
    for i, b in enumerate(batch):
        # Consider all the proteins in the batch
        x = np.stack([b['coords'][c] for c in ['N', 'CA', 'C', 'CB', 'O']], 1)

        l = b['length']
        # x : [l, 4, 3]
        x_pad = np.pad(x, [[0, L_max-l], [0, 0], [0, 0]],
                       'constant', constant_values=(np.nan, ))
        # x_pad : [L_max, 4, 4]
        X[i, :, :, :] = x_pad

        # X : [Batch, L_max , 4, 3]
        # X[i] : [L_max , 4, 3]

        # Convert sequences to labels
        indices_aa = np.asarray([alphabet.index(a)
                             for a in b['seq']], dtype=np.int32)
        indeices_cctop = np.asarray([cctop_code.index(a)
                             for a in b['cctop']], dtype=np.int32)
        if shuffle_fraction > 0.:
            idx_shuffle = shuffle_subset(l, shuffle_fraction)
            S[i, :l] = indices_aa[idx_shuffle]
            C[i, :l] = indeices_cctop[idx_shuffle]
        else:
            S[i, :l] = indices_aa
            C[i, :l] = indeices_cctop
    # Mask
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    # S为一个batch的标签, S : [Batch,Length_max] (0,19 int)对应每个位置的氨基酸类型,注意mask为True的0代表Alanine,mask为Falfse的0代表这个序列长度没有这么长，没有这个氨基酸
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    bernoulli_mask = torch.rand(S.shape,device=device)
    # mask_fraction_matrix = torch.full(S.shape,mask_fraction,device=device)
    # S_mask = torch.where( bernoulli_mask< mask_fraction, torch.tensor(20,dtype=torch.long,device=device) ,S)

    mask_fraction = torch.as_tensor(np.random.uniform(mask_min,mask_fraction,B),device=device)
    mask_fraction = mask_fraction.unsqueeze(-1)
    S_mask = torch.where(bernoulli_mask< mask_fraction, torch.tensor(20,dtype=torch.long,device=device) ,S)

    C = torch.from_numpy(C).to(dtype=torch.long, device=device)
    # X为一个batch的数据, X : [Batch,Length_max,4,3] (float32)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    # mask为一个batch的mask标签, mask : [Batch, Length_max] (0,1 float32)
    mask = torch.from_numpy(mask).to(
        dtype=torch.float32, device=device)
    # Length 为一个列表，长度为batch，存储这一个batch里蛋白质的长度
    return X, S, C,mask, lengths, S_mask


def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    # S.view(-1) [B*L,]
    # log_probs.view(-1,log_probs.size(-1)) [B*L,20]
    # loss [B,L]
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


def loss_cse(S, logits, mask, smooth=0.0):
    """Cross Entropy Loss with mask"""
    # Logits : [Batch, Length_max, 20]   float
    # S      : [Batch, Length]           long
    # Mask   : [Batch, Length]           0/1
    criterion = torch.nn.CrossEntropyLoss(
        reduction="none", label_smoothing=smooth)
    loss = criterion(
        logits.reshape(-1, logits.shape[-1]), S.reshape(-1)).reshape(S.shape)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def loss_smoothed(S, log_probs, mask, weight=0.1,num_classes=20):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S,num_classes).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss,loss_av

def backbone_select(pdb_structure: Union[str, Bio.PDB.Structure.Structure], output_path, atom_select=4):
    parser = PDBParser()
    if isinstance(pdb_structure, str):
        if not os.path.exists(pdb_structure):
            raise FileNotFoundError(f"Not exists {pdb_structure}")
        pdb_structure = parser.get_structure("test", pdb_structure)

    class BackboneSelect(Select):
        def accept_residue(self, residue):
            if residue.get_id()[0] != " ":  # exclude other chains
                return 0
            else:
                return 1

        def accept_atom(self, atom):
            if atom_select == 1:
                if atom.get_ed() == "CA":
                    return 1
                else:
                    return 0
            else:
                if atom.get_id() == "N":
                    return 1
                elif atom.get_id() == "CA":
                    return 1
                elif atom.get_id() == "C":
                    return 1
                elif atom.get_id() == "O" and atom_select == 4:
                    return 1
                else:
                    return 0
    io = PDBIO()
    io.set_structure(pdb_structure)
    io.save(output_path, BackboneSelect())


def load_checkpoint(checkpoint_path, model,device="cpu"):
    print('Loading checkpoint from {}'.format(checkpoint_path))
    state_dicts = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dicts['model_state_dict'])
    print('\tEpoch {}'.format(state_dicts['epoch']))
    return



def gaussian(x, std):
    pi = torch.tensor(torch.pi)
    s2 = 2.0*torch.tensor(std).square()
    x2 = torch.tensor(x).square().neg()

    return torch.exp(x2 / s2) * torch.rsqrt(s2 * pi)


def gaussian_kernel(kernel_size, std=1.0):
    kernel = [gaussian(i - (kernel_size // 2), std)
              for i in range(kernel_size)]

    kernel = torch.tensor(kernel)
    kernel = kernel / kernel.sum()

    return kernel


class CoordBatchConverter(BatchConverter):
    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch: Sequence[Tuple[Sequence, str]], device=None):
        """
        Args:
            raw_batch: List of tuples (coords, confidence, seq)
            In each tuple,
                coords: list of floats, shape L x 3 x 3
                confidence: list of floats, shape L; or scalar float; or None
                seq: string of length L
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        self.alphabet.cls_idx = self.alphabet.get_idx("<cath>")
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = 'X' * len(coords)
            batch.append(((coords, confidence), seq))

        coords_and_confidence, strs, tokens = super().__call__(batch)

        # pad beginning and end of each protein due to legacy reasons
        coords = [
            F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.inf)
            for cd, _ in coords_and_confidence
        ]
        confidence = [
            F.pad(torch.tensor(cf), (1, 1), value=-1.)
            for _, cf in coords_and_confidence
        ]
        coords = self.collate_dense_tensors(coords, pad_v=np.nan)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1.)
        if device is not None:
            coords = coords.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
        padding_mask = torch.isnan(coords[:, :, 0, 0])
        coord_mask = torch.isfinite(coords.sum(-2).sum(-1))
        confidence = confidence * coord_mask + (-1.) * padding_mask
        return coords, confidence, strs, tokens, padding_mask

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        """
        Args:
            coords_list: list of length batch_size, each item is a list of
            floats in shape L x 3 x 3 to describe a backbone
            confidence_list: one of
                - None, default to highest confidence
                - list of length batch_size, each item is a scalar
                - list of length batch_size, each item is a list of floats of
                    length L to describe the confidence scores for the backbone
                    with values between 0. and 1.
            seq_list: either None or a list of strings
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v):
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
        # assumes all on same device
        (device,) = tuple(set(x.device for x in samples))
        max_shape = [max(lst) for lst in zip(
            *[x.shape for x in samples])]  # 必须要zip打包 zip (*)
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]  # 浅拷贝
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result

def rank_jsonl(input_file, output_file):
    data = []
    with open(input_file, "r") as f:
        for line in f:
            # 每一行代表一个序列的字典字符串加一个'\n',对于字符串应该用json.loads()文件进行读取
            data.append(json.loads(line))
    data = sorted(data, key=lambda pro: pro['length'])
    with open(output_file, "w") as f:
        for p in data:
            f.write(json.dumps(p))
# rank_jsonl("/data/home/scv6707/run/zxt/data/TMfromzb/json/content.jsonl", "./jsonl")
import torch
from openfold.utils.rigid_utils import Rotation, Rigid
def get_bb_frames(coords):
    """
    Returns a local rotation frame defined by N, CA, C positions.
    Args:
        coords: coordinates, tensor of shape (batch_size x length x 3 x 3)
        where the third dimension is in order of N, CA, C
    Returns:
        Local relative rotation frames in shape (batch_size x length x 3 x 3)
        Local translation in shape (batch_size x length x 3)
    """
    v1 = coords[:, :, 2] - coords[:, :, 1]
    v2 = coords[:, :, 0] - coords[:, :, 1]
    e1 = normalize(v1, dim=-1)  # [B, L, 3]
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = normalize(u2, dim=-1)  # [B, L, 3]
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)
    t = coords[:, :, 1]  # translation is just the CA atom coordinate
    return Rigid(Rotation(R), t)

def norm(tensor, dim, eps=1e-8, keepdim=False):
    """
    Returns L2 norm along a dimension.
    """
    return torch.sqrt(
        torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)


def normalize(tensor, dim=-1):
    """
    Normalizes a tensor along a dimension after removing nans.
    """
    return nan_to_num(
        torch.div(tensor, norm(tensor, dim=dim, keepdim=True))
    )


def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.    
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)