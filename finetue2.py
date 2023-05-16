import os
import sys
import typing as T
from dataclasses import dataclass
from tqdm import *
import torch
import torch.nn as nn
from omegaconf import MISSING
from openfold.data.data_transforms import make_atom14_masks
from openfold.utils.rigid_utils import Rigid
from openfold.np import residue_constants
from openfold.utils.loss import compute_predicted_aligned_error, compute_tm, compute_fape
from torch import nn
from torch.nn import LayerNorm
from pathlib import Path
from esm.myattention import AttentionMap
import esm
from esm import Alphabet
from esm.esmfold.v1.categorical_mixture import categorical_lddt
from esm.esmfold.v1.trunk import FoldingTrunk, FoldingTrunkConfig
from esm.esmfold.v1.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
    output_to_pdb,
)
from esm.data import read_fasta
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument(
        "-i",
        "--fasta",
        help="Path to input FASTA file",
        type=Path,
        required=True,
    ) 
parser.add_argument(
        "-o", "--pdb", help="Path to output PDB directory", type=Path, required=True
    )
parser.add_argument(
        "--num-recycles",
        type=int,
        default=None,
        help="Number of recycles to run. Defaults to number used in training (4).",
    )
parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=500,
        help="Maximum number of tokens per gpu forward-pass. This will group shorter sequences together "
        "for batched prediction. Lowering this can help with out of memory issues, if these occur on "
        "short sequences.",
    )
parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunks axial attention computation to reduce memory usage from O(L^2) to O(L). "
        "Equivalent to running a for loop over chunks of of each dimension. Lower values will "
        "result in lower memory usage at the cost of speed. Recommended values: 128, 64, 32. "
        "Default: None."
    )
parser.add_argument("--cpu-only", help="CPU only", action="store_true")
parser.add_argument(
        "--cpu-offload", help="Enable CPU offloading", action="store_true"
    )
args = parser.parse_args()
sys.path.append(sys.path[0][0:-6] +'/TMbed_fromxu/tmbed')
from predict import tmbed


@dataclass
class ESMFoldConfig:
    trunk: T.Any = FoldingTrunkConfig()
    lddt_head_hid_dim: int = 128

class ESMFold(nn.Module):
    def __init__(self, esmfold_config=None, **kwargs):
        super().__init__()

        self.cfg = esmfold_config if esmfold_config else ESMFoldConfig(**kwargs)
        cfg = self.cfg

        self.distogram_bins = 64
        
        self.tmbed_model = tmbed()


        self.esm, self.esm_dict = esm.pretrained.esm2_t36_3B_UR50D()

        self.esm.requires_grad_(False)
        self.esm.half() # ???
        # self.esm_feats,2560; 
        self.esm_feats = self.esm.embed_dim
        # self.esm_attns=1440; self.esm.num_layers=36; self.esm.attention_heads=40
        self.esm_attns = self.esm.num_layers * self.esm.attention_heads
        self.register_buffer("af2_to_esm", ESMFold._af2_to_esm(self.esm_dict))
        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm.num_layers + 1))
        # c_s=1024, c_z=128

        c_s = cfg.trunk.sequence_state_dim
        c_z = cfg.trunk.pairwise_state_dim

        self.esm_s_mlp = nn.Sequential(
            LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )
        self.tm_dim = 5
        # 3,65,5 -> 3,65,2560 -> 3,65,37*2560
        self.tmbed2s = nn.Sequential(
            # LayerNorm(self.esm_feats),
            nn.Linear(self.tm_dim, self.esm_feats, bias=False),
            nn.ReLU(),
            nn.Linear(self.esm_feats, (self.esm.num_layers+1)*self.esm_feats, bias=False),
        )
        self.tmbed2z = AttentionMap(self.tm_dim, self.esm.embed_dim, num_heads=128, gated=False)

        # 0 is padding, N is unknown residues, N + 1 is mask.
        self.n_tokens_embed = residue_constants.restype_num + 3
        self.pad_idx = 0
        self.unk_idx = self.n_tokens_embed - 2
        self.mask_idx = self.n_tokens_embed - 1
        self.embedding = nn.Embedding(self.n_tokens_embed, c_s, padding_idx=0)

        self.trunk = FoldingTrunk(**cfg.trunk)

        self.distogram_head = nn.Linear(c_z, self.distogram_bins)
        self.ptm_head = nn.Linear(c_z, self.distogram_bins)
        self.lm_head = nn.Linear(c_s, self.n_tokens_embed)
        self.lddt_bins = 50
        self.lddt_head = nn.Sequential(
            nn.LayerNorm(cfg.trunk.structure_module.c_s),
            nn.Linear(cfg.trunk.structure_module.c_s, cfg.lddt_head_hid_dim),
            nn.Linear(cfg.lddt_head_hid_dim, cfg.lddt_head_hid_dim),
            nn.Linear(cfg.lddt_head_hid_dim, 37 * self.lddt_bins),
        )

    @staticmethod
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [
            d.get_idx(v) for v in residue_constants.restypes_with_x
        ]
        return torch.tensor(esm_reorder)

    def _af2_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return self.af2_to_esm[aa]

    def _compute_language_model_representations(
        self, esmaa: torch.Tensor
    ) -> torch.Tensor:
        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        batch_size = esmaa.size(0)

        bosi, eosi = self.esm_dict.cls_idx, self.esm_dict.eos_idx
        bos = esmaa.new_full((batch_size, 1), bosi)
        eos = esmaa.new_full((batch_size, 1), self.esm_dict.padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

        res = self.esm(
            esmaa,
            repr_layers=range(self.esm.num_layers + 1),
            need_head_weights=False,
        )
        esm_s = torch.stack(
            [v for _, v in sorted(res["representations"].items())], dim=2
        )
        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
        return esm_s

    def _mask_inputs_to_esm(self, esmaa, pattern):
        new_esmaa = esmaa.clone()
        new_esmaa[pattern == 1] = self.esm_dict.mask_idx
        return new_esmaa

 
    def forward(
        self,
        # aa: torch.Tensor,
        # mask: T.Optional[torch.Tensor] = None,
        # residx: T.Optional[torch.Tensor] = None,
        # masking_pattern: T.Optional[torch.Tensor] = None,
        # num_recycles: T.Optional[int] = None,
        # seq= None,
        sequences: T.Union[str, T.List[str]],
        residx=None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = None,
        residue_index_offset: T.Optional[int] = 512,
        chain_linker: T.Optional[str] = "G" * 25,
    ):
        
        """Runs a forward pass given input tokens. Use `model.infer` to
        run inference from a sequence.

        Args:
            aa (torch.Tensor): Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            mask (torch.Tensor): Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
        """  
        with torch.no_grad():      
            if isinstance(sequences, str):
                sequences = [sequences]

            aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
                sequences, residue_index_offset, chain_linker
            )

            if residx is None:
                residx = _residx
            elif not isinstance(residx, torch.Tensor):
                residx = collate_dense_tensors(residx)

            aatype, mask, residx, linker_mask = map(
                lambda x: x.to(self.device), (aatype, mask, residx, linker_mask)
            )
            aa = aatype
            
            if mask is None:
                mask = torch.ones_like(aa)

            B = aa.shape[0] # batchsize
            L = aa.shape[1] # L是全序列最大值,不足用0补充
            device = aa.device

            if residx is None:
                residx = torch.arange(L, device=device).expand_as(aa)

        # === ESM ===
            esmaa = self._af2_idx_to_esm_idx(aa, mask)

            if masking_pattern is not None:
                esmaa = self._mask_inputs_to_esm(esmaa, masking_pattern)
        # esm_s应是语言模型抽取的embedding，预测个数×残基数×37×2560;
        # 其中,第二维(dim=1维)的前len+1有数,其余mask为0.即[i,0:len+1,:,:]有数字???
        # ???为什么会多一位非零位
        esm_s = self._compute_language_model_representations(esmaa)
        # esm_s = esm_s.detach()
        # tmbed_s为B, ml, 5,其中[i,0:len,:,:]有值,其余mask
        tmbed_s = self.tmbed_model(sequences).to(device)
        # tmbed_s = tmbed_s.detach()
        esm_s += self.tmbed2s(tmbed_s).reshape(B, L, (self.esm.num_layers+1), self.esm.embed_dim)       # b,ml, 37*2560->b,ml,37,2560


        # Convert esm_s to the precision used by the trunk and
        # the structure module. These tensors may be a lower precision if, for example,
        # we're running the language model in fp16 precision.类似于浮点类型转换。转换使plm于后续模块精度一致
        esm_s = esm_s.to(self.esm_s_combine.dtype)

        # esm_s = esm_s.detach()

        # === preprocessing ===预测个数×残基数×2560;1×65×2560
        # s为esm_s, layer_weights即esm_s_combine，=nn.Parameter(torch.zeros(self.esm.num_layers + 1))
        # s = (softmax(layer_weights) * s).sum(0) esm_s_combine作为权重，将37层做加权和
        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        # 1×65×1024
        s_s_0 = self.esm_s_mlp(esm_s)
        # 1×65×65×1024
        s_z_0 = self.tmbed2z(tmbed_s, mask=mask)
        # s_z_0 = s_s_0.new_zeros(B, L, L, self.cfg.trunk.pairwise_state_dim)
        # esm生成序列信息加上embedding？
        s_s_0 += self.embedding(aa)

        structure: dict = self.trunk(
            s_s_0, s_z_0, aa, residx, mask, no_recycles=num_recycles,
        )
        # Documenting what we expect:
        structure = {
            k: v
            for k, v in structure.items()
            if k
            in [
                "s_z",
                "s_s",
                "frames",
                "sidechain_frames",
                "unnormalized_angles",
                "angles",
                "positions",
                "states",
            ]
        }

        with torch.no_grad():
            # 8,b,ml,14,3 -> 8,b,ml,3,3 -> 
            structure['backbone_positions'] = structure['positions'][..., :3, :]

            # structure["s_z"]为b,ml,ml,128 ->linear b,ml,ml,64 
            disto_logits = self.distogram_head(structure["s_z"])
            disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
            structure["distogram_logits"] = disto_logits
            # structure["s_s"]为b,ml,1024->(linear) b,ml,23        
            lm_logits = self.lm_head(structure["s_s"])
            structure["lm_logits"] = lm_logits

            structure["aatype"] = aa # b,ml
            make_atom14_masks(structure)

            for k in [
                "atom14_atom_exists", # b,ml,14, 为0，1值.表征14原子表示下，哪写原子存在(置为1
                "atom37_atom_exists",  # b,ml,37
            ]:
                structure[k] *= mask.unsqueeze(-1)
            structure["residue_index"] = residx #b,nml
            # structure["states"]为8,b,lm,384->8,b,lm,37,50
            lddt_head = self.lddt_head(structure["states"]).reshape(
                structure["states"].shape[0], B, L, -1, self.lddt_bins
            )
            structure["lddt_head"] = lddt_head 
            # 取lddt_head最后一维，其维度为(b,lm,37,50)生成plddt(b,ml,37)
            plddt = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)
            structure["plddt"] = 100 * plddt  # we predict plDDT between 0 and 1, scale to be between 0 and 100.
            # structure["s_z"]为b,ml,ml,128 ->linear b,ml,ml,64
            ptm_logits = self.ptm_head(structure["s_z"])

            seqlen = mask.type(torch.int64).sum(1) #每个序列长度的tensor
            structure["ptm_logits"] = ptm_logits
            structure["ptm"] = torch.stack([
                compute_tm(batch_ptm_logits[None, :sl, :sl], max_bins=31, no_bins=self.distogram_bins)
                for batch_ptm_logits, sl in zip(ptm_logits, seqlen)
            ]) # torch.Size([2])
            # 增加了'aligned_confidence_probs', 'predicted_aligned_error', 'max_predicted_aligned_error'
            structure.update(
                compute_predicted_aligned_error(
                    ptm_logits, max_bin=31, no_bins=self.distogram_bins
                )
            )
            # b,ml,37 * b,ml,1
            structure["atom37_atom_exists"] = structure[
                "atom37_atom_exists"
            ] * linker_mask.unsqueeze(2)
            # plddt为b,ml,37, 求每个蛋白的全氨基酸全37原子的平均plddt,一个蛋白得到一个值
            structure["mean_plddt"] = (structure["plddt"] * structure["atom37_atom_exists"]).sum(
                dim=(1, 2)
            ) / structure["atom37_atom_exists"].sum(dim=(1, 2))
            structure["chain_index"] = chain_index
            structure['frame_mask'] = mask
            structure['backbone_atoms_mask'] = mask.repeat_interleave(3, dim=-1)

        return structure

    def output_to_pdb(self, output: T.Dict) -> T.List[str]:
        """Returns the pbd (file) string from the model given the model output."""
        return output_to_pdb(output)

    def infer_pdbs(self, seqs: T.List[str], *args, **kwargs) -> T.List[str]:
        """Returns list of pdb (files) strings from the model given a list of input sequences."""
        output = self.infer(seqs, *args, **kwargs)
        return self.output_to_pdb(output)

    def infer_pdb(self, sequence: str, *args, **kwargs) -> str:
        """Returns the pdb (file) string from the model given an input sequence."""
        return self.infer_pdbs([sequence], *args, **kwargs)[0]

    def set_chunk_size(self, chunk_size: T.Optional[int]):
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-lengthed chunks.
        # Setting the value to None will return to default behavior, disable chunking.
        self.trunk.set_chunk_size(chunk_size)

    @property
    def device(self):
        return self.esm_s_combine.device
# import urllib
def _load_model(model_name):
    if model_name.endswith(".pt"):  # local, treat as filepath
        model_path = Path(model_name)
        model_data = torch.load(str(model_path), map_location="cpu")
    else:  # load from hub
        
        url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
        # urllib.request.urlretrieve(url, filename='esmfold_3B_v1.pt')
        model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")

    cfg = model_data["cfg"]["model"]
    model_state = model_data["model"]
    model = ESMFold(esmfold_config=cfg)

    # expected_keys = set(model.state_dict().keys())
    # found_keys = set(model_state.keys())

    # missing_essential_keys = []
    # for missing_key in expected_keys - found_keys:
    #     if not missing_key.startswith("esm."):
    #         missing_essential_keys.append(missing_key)

    # if missing_essential_keys:
    #     raise RuntimeError(f"Keys '{', '.join(missing_essential_keys)}' are missing.")

    model.load_state_dict(model_state, strict=False)

    return model

def create_batched_sequence_datasest(
    file_path, max_tokens_per_batch: int = 1024
): # 能不能实现shuffle，采用随机for循环？
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

all_sequences = sorted(
        read_fasta(args.fasta), key=lambda header_seq: len(header_seq[1])
    )
args.pdb.mkdir(exist_ok=True)

def generate_label(structures, device='cuda'):
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

    N_pos_tensor = collate_dense_tensors(N_pos_list).to(device)
    CA_pos_tensor = collate_dense_tensors(CA_pos_list).to(device)
    C_pos_tensor = collate_dense_tensors(C_pos_list).to(device)
    target_positions = collate_dense_tensors(all_pos_list).to(device)
    target_frames =  Rigid.from_3_points(N_pos_tensor, CA_pos_tensor, C_pos_tensor)

    return target_positions, target_frames


tm_dataloader = create_batched_sequence_datasest('./data/TMfromzb/json/content.jsonl', max_tokens_per_batch=1024)

loss = compute_fape
# with torch.no_grad():
#     model = _load_model("esmfold_3B_v1.pt")
# model = _load_model("esmfold_3B_v1.pt")
def finetune_esm(net, loss, train_dataloader, device='cuda', batch_size=2, num_epoch=1, lr=0.01):

    for epoch in range(num_epoch):

        print("——————第 {} 轮训练开始——————".format(epoch + 1))
        # 训练开始
        # net.train()
        train_acc = 0
        for batch in tqdm(train_dataloader, desc='训练'):
            headers, sequences, structures = batch #list装东西
            # list不能直接放到GPU，后面把tensor放过去;sequences = sequences.to(device); structures = structures.to(device); headers = headers.to(device) 
            target_positions, target_frames = generate_label(structures)
            output_dict = net(sequences) #header放在哪

            # finetune
            params_1x, params_10x = [], []
            for name, param in net.named_parameters():
                if param.requires_grad:
                    if name in ['tmbed2s.0.weight', 'tmbed2s.2.weight',
                        'tmbed2z.dimensionup.weight','tmbed2z.proj.weight','tmbed2z.o_proj.weight','tmbed2z.o_proj.bias']:
                        params_10x.append(param)
                    else:
                        params_1x.append(param)

            # params_1x = [param for name, param in net.named_parameters() if name not in ['tmbed2s.0.weight', 'tmbed2s.2.weight',
            #         'tmbed2z.dimensionup.weight','tmbed2z.proj.weight','tmbed2z.o_proj.weight','tmbed2z.o_proj.bias']]

            optimizer = torch.optim.SGD([{'params': params_1x},
                                   {'params': params_10x, 'lr': lr * 10}], lr=lr)
            




            # ->b,ml,7->rigid
            pred_frames = Rigid.from_tensor_7(output_dict['frames'][-1, ...])
            # 8,b,ml,3,3->b,ml,3,3->b,3,ml,3->b,3*ml,3
            pred_positions = torch.flatten(output_dict['backbone_positions'][-1, ...].permute(0,2,1,3), start_dim=-3, end_dim=-2)
            frames_mask = output_dict['frame_mask']; positions_mask = output_dict['backbone_atoms_mask']

            Loss = loss(
                        pred_frames=pred_frames,
                        target_frames=target_frames,
                        frames_mask=frames_mask,
                        pred_positions=pred_positions,
                        target_positions=target_positions,
                        positions_mask=positions_mask,
                        length_scale=10,
                        )
          
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

# model = esm.pretrained.esmfold_v1()
model = _load_model("esmfold_3B_v1")
net = model.eval()
# net = model.train().cuda()
net.set_chunk_size(args.chunk_size)
net.cuda()

loss = compute_fape


finetune_esm(net, loss, train_dataloader=tm_dataloader)
# for headers, sequences, structures in train_dataloader:
#     # tm = tmbed()
#     # a = tm(sequences)
#     output = model(sequences, num_recycles=args.num_recycles)
#     # out_format控制输出概率还是分类
#     output = model.infer(sequences, num_recycles=args.num_recycles)


