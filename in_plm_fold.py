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
from torch.utils.checkpoint import checkpoint
from pathlib import Path
from esm.myattention import AttentionMap
import esm
from esm import Alphabet
from esm.esmfold.v1.categorical_mixture import categorical_lddt
from esm.esmfold.v1.trunk import FoldingTrunk, FoldingTrunkConfig
from esm.esmfold.v1.misc import output_to_pdb
from torchcrf import CRF
from esm.esmfold.v1.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
    output_to_pdb,
)


def load_plm(esm2_path, ft_path):
        from esm.pretrained import load_regression_hub, load_model_and_alphabet_core
        model_data = torch.load(esm2_path)
        update_data = torch.load(ft_path)
        ft_model_state = update_data['model_state_dict']
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


class ESMFoldConfig:
    trunk: T.Any = FoldingTrunkConfig()
    lddt_head_hid_dim: int = 128

class ESMFold(nn.Module):
    def __init__(self, pattern='no', esmfold_config=None, add_tmbed=False, num_tags=6, **kwargs, ):
        super().__init__()

        self.cfg = esmfold_config if esmfold_config else ESMFoldConfig(**kwargs)
        cfg = self.cfg
        self.add_tmbed = add_tmbed
        self.distogram_bins = 64


        # self.esm, self.esm_dict = esm.pretrained.esm2_t36_3B_UR50D()
        self.esm_dict = esm.data.Alphabet.from_architecture("ESM-1b")
        # self.esm.requires_grad_(False)
        # self.esm.half() # ???
        self.esm_feats=2560; self.num_layers=36
        # self.esm_feats = self.esm.embed_dim
        # self.esm_attns=1440; self.esm.num_layers=36; self.esm.attention_heads=40
        # self.esm_attns = self.esm.num_layers * self.esm.attention_heads
        self.register_buffer("af2_to_esm", ESMFold._af2_to_esm(self.esm_dict))
        # self.esm_s_combine = nn.Parameter(torch.zeros(self.num_layers + 1))
        # c_s=1024, c_z=128

        c_s = cfg.trunk.sequence_state_dim
        c_z = cfg.trunk.pairwise_state_dim
        self.num_tags = num_tags
        self.crf = CRF(self.num_tags,batch_first=True)

        self.esm_s_mlp = nn.Sequential(
            LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )

        if add_tmbed:
            from tmbed.predict import tmbed2
            self.tmbed_model = tmbed2(use_gpu=False)
            self.tm_dim = 5
            self.tm_s_mlp = nn.Sequential(
                nn.Linear(self.tm_dim, c_s),
                nn.ReLU(),
                LayerNorm(c_s),
                nn.Linear(c_s, c_s),
            )

        # 3,65,5 -> 3,65,2560 -> 3,65,37*2560
        # self.tmbed2s = nn.Sequential(
        #     # LayerNorm(self.esm_feats),
        #     nn.Linear(self.tm_dim, self.esm_feats, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(self.esm_feats, (self.esm.num_layers+1)*self.esm_feats, bias=False),
        # )
        # self.tmbed2z = AttentionMap(self.tm_dim, self.esm.embed_dim, num_heads=128, gated=False)

        # 0 is padding, N is unknown residues, N + 1 is mask.
        self.n_tokens_embed = residue_constants.restype_num + 3
        self.pad_idx = 0
        self.unk_idx = self.n_tokens_embed - 2
        self.mask_idx = self.n_tokens_embed - 1
        self.embedding = nn.Embedding(self.n_tokens_embed, c_s, padding_idx=0)

        self.trunk = FoldingTrunk(self.num_tags, **cfg.trunk)

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
        self._frozen(pattern=pattern)

    def _frozen(self,pattern="no"):
        """
        Only training the following part of the modules:
        - ipa module
        - bb updata
        - transition
        - 
        - 
        """
        if pattern == "no":
            for name, parameter in self.named_parameters():
                # if name.startswith("tm_s"):
                    #  parameter.requires_grad = True
                # elif name.startswith("esm_s_"):
                #     parameter.requires_grad = True
                if name.startswith("trunk.structure_module"):
                     parameter.requires_grad = True
                # elif name.startswith("embedding"):
                #     parameter.requires_grad = True
                else:
                    parameter.requires_grad = False
        
        if pattern == "withseq":
            for name,parameter in self.named_parameters():
                if name.startswith("tm_s"):
                    parameter.requires_grad = True
                # elif name.startswith("esm_s_"):
                #     parameter.requires_grad = True
                elif name.startswith("trunk.structure_module"):
                    parameter.requires_grad = True
                # elif name.startswith("embedding"):
                #     parameter.requires_grad = True
                elif name.startswith("trunk.cctop"):
                    parameter.requires_grad = True
                else:
                    parameter.requires_grad = False
        if pattern == "all":
            for name,parameter in self.named_parameters():
                if name.startswith("tmbed"):
                    parameter.requires_grad = False
                elif name.startswith("esm.layers."):
                    parameter.requires_grad = False
                else: 
                    parameter.requires_grad = True
   
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
        esm_s,
        aa, 
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = None,
        # seq= None,
        
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
        
        # if isinstance(sequences, str):
        #     sequences = [sequences]

        # aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
        #     sequences, residue_index_offset, chain_linker
        # )

        # if residx is None:
        #     residx = _residx
        # elif not isinstance(residx, torch.Tensor):
        #     residx = collate_dense_tensors(residx)

        # aa, mask, residx = map(
        #     lambda x: x.to(self.device), (aa, mask, residx)
        # )
        # aa = aatype
        # if not isinstance(esm_s, torch.Tensor):
            

        #     aa, mask, residx, linker_mask, chain_index = batch_encode_sequences(
        #         aa
        #         # , residue_index_offset, chain_linker
        #     )
        #     aa, mask, residx, linker_mask = map(
        #     lambda x: x.to(self.device), (aa, mask, residx, linker_mask)
        # )
        # if mask is None:
        #     mask = torch.ones_like(aa)

        B = esm_s.shape[0] # batchsize
        L = esm_s.shape[1] # L是全序列最大值,不足用0补充
        device = esm_s.device

        # if residx is None:
        #     residx = torch.arange(L, device=device).expand_as(aa)

        # === ESM ===
        # esmaa=aa
        # esmaa = self._af2_idx_to_esm_idx(aa, mask)

        # if masking_pattern is not None:
        #         esmaa = self._mask_inputs_to_esm(esmaa, masking_pattern)
        # esm_s应是语言模型抽取的embedding，预测个数×残基数×37×2560;
        # 其中,第二维(dim=1维)的前len+1有数,其余mask为0.即[i,0:len+1,:,:]有数字???
        # ???为什么会多一位非零位
        # esm_s = self._compute_language_model_representations(esmaa)
        # esm_s = esm_s.detach()
        
        # tmbed_s为B, ml, 5,其中[i,0:len,:,:]有值,其余mask
        if self.add_tmbed:
            tmbed_embedding = self.tmbed_model(aatype=aa, mask=mask).to(device)
            tmbed_embedding = tmbed_embedding.detach()
            tmbed_s = self.tm_s_mlp(tmbed_embedding)
        else:
            tmbed_s = None


        # Convert esm_s to the precision used by the trunk and
        # the structure module. These tensors may be a lower precision if, for example,
        # we're running the language model in fp16 precision.类似于浮点类型转换。转换使plm于后续模块精度一致
        # esm_s = esm_s.to(self.esm_s_combine.dtype)

        # esm_s = esm_s.detach()

        # === preprocessing ===预测个数×残基数×2560;1×65×37x2560
        # s为esm_s, layer_weights即esm_s_combine，=nn.Parameter(torch.zeros(self.esm.num_layers + 1))
        # s = (softmax(layer_weights) * s).sum(0) esm_s_combine作为权重，将37层做加权和
        # esm_s = esmaa[:,:,None,None].repeat(1,1,37,2560)+0.0 #cuda不支持整数的矩阵乘法
        # esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        # 1×65×1024
        s_s_0 = self.esm_s_mlp(esm_s)
        # 1×65×65×1024
        # s_z_0 = self.tmbed2z(tmbed_s, mask=mask)
        s_z_0 = s_s_0.new_zeros(B, L, L, self.cfg.trunk.pairwise_state_dim)
        # esm生成序列信息加上embedding？
        s_s_0 += self.embedding(aa)
        # num_recycles=1
        structure: dict = self.trunk(s_s_0, s_z_0, aa, residx, mask, add_tmbed=self.add_tmbed, tm=tmbed_s, no_recycles=num_recycles)
        # structure: dict = checkpoint(self.trunk, s_s_0, s_z_0, aa, residx, mask,)

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
                "logits_cctop"
            ]
        }

        

        # # 8,b,ml,14,3 -> 8,b,ml,3,3 -> 
        # structure['backbone_positions'] = structure['positions'][..., :3, :]
        # # 8,b,ml,3,3->b,ml,3,3->b,3,ml,3->b,3*ml,3
        # structure['backbone_positions']  = torch.flatten(structure['backbone_positions'][-1, ...].permute(0,2,1,3), start_dim=-3, end_dim=-2)
        structure['backbone_positions'] = torch.flatten(
            structure['positions'][-1, :,:, :3, :].permute(0,2,1,3),  
            start_dim=-3, end_dim=-2)

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
        # structure["atom37_atom_exists"] = structure[
        #     "atom37_atom_exists"
        # ] * linker_mask.unsqueeze(2)
        # plddt为b,ml,37, 求每个蛋白的全氨基酸全37原子的平均plddt,一个蛋白得到一个值
        structure["mean_plddt"] = (structure["plddt"] * structure["atom37_atom_exists"]).sum(
            dim=(1, 2)
        ) / structure["atom37_atom_exists"].sum(dim=(1, 2))
        # structure["chain_index"] = chain_index
        structure['frame_mask'] = mask
        structure['backbone_atoms_mask'] = mask.repeat_interleave(3, dim=-1)
        structure['pred_frames'] = Rigid.from_tensor_7(structure['frames'][-1, ...])


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
    
    def decode_crf(self,emission,mask):
        """
        CRF decode the sequence
        """
        if not isinstance(mask.dtype,torch.ByteTensor):
            return self.crf.decode(emission,mask=mask.byte())
        else:
            return self.crf.decode(emission,mask=mask)

    def neg_loss_crf(self,emission,tag,mask):
        """
        CRF score for the cctop
        Input 
        - emission  [B, N, C] (batch_first = True)
        - tag       [B, N]
        - mask      [B, N]
        output : 
        scaler
        """
        if not isinstance(mask.dtype,torch.ByteTensor):
            return (self.crf(emission, tag, mask=mask.byte(),reduction="token_mean")).neg()
        else:
            return (self.crf(emission, tag, mask=mask,reduction="token_mean")).neg()
  
      

    @property
    def device(self):
        return self.esm_s_combine.device


def load_model(chunk_size, pattern='all', num_tags=6, add_tmbed=False, model_path="/pubhome/bozhang/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt", cfg=None):

    
    model_data = torch.load(str(model_path), map_location="cpu") #读取一个pickle文件为一个dict
    if cfg == None:
        cfg = model_data["cfg"]["model"]
    model = ESMFold(pattern=pattern, num_tags=num_tags, add_tmbed=add_tmbed, esmfold_config=cfg) # make an instance
    model_state = model_data["model"]
    model.load_state_dict(model_state, strict=False)
    model.set_chunk_size(chunk_size)
    return model