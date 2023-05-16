from pathlib import Path
import sys
import logging
import typing as T
import argparse
import omegaconf
from TPFold import ESMFold, load_model_old
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8"
import torch
from openfold.utils.feats import atom14_to_atom37
from openfold.np.protein import Protein as OFProtein
from openfold.np.protein import to_pdb

import esm
from esm.data import read_fasta
from timeit import default_timer as timer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S",
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


PathLike = T.Union[str, Path]


def enable_cpu_offloading(model):
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import enable_wrap, wrap

    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://localhost:9999", world_size=1, rank=0
    )

    wrapper_kwargs = dict(cpu_offload=CPUOffload(offload_params=True))

    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
        for layer_name, layer in model.layers.named_children():
            wrapped_layer = wrap(layer)
            setattr(model.layers, layer_name, wrapped_layer)
        model = wrap(model)

    return model


def init_model_on_gpu_with_cpu_offloading(model):
    model = model.eval()
    model_esm = enable_cpu_offloading(model.esm)
    del model.esm
    model.cuda()
    model.esm = model_esm
    return model


def output_to_pdb(output: T.Dict, final_atom_mask=None) -> T.List[str]:
    """Returns the pbd (file) string from the model given the model output."""
    # atom14_to_atom37 must be called first, as it fails on latest numpy if the
    # input is a numpy array. It will work if the input is a torch tensor.
    final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
    output = {k: v.to("cpu").numpy() for k, v in output.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    if final_atom_mask == None:
        final_atom_mask = output["atom37_atom_exists"]
    pdbs = []
    for i in range(output["aatype"].shape[0]):
        aa = output["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = output["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=output["plddt"][i],
            chain_index=output["chain_index"][i] if "chain_index" in output else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def create_batched_sequence_datasest(
    sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:

    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)

    yield batch_headers, batch_sequences


if __name__ == "__main__":
    print(f"cuda_avial:{torch.cuda.is_available()}")
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
        default=1024,
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
    # parser.add_argument(
    #     "--parameters", type=str, default="/lustre/gst/xuchunfu/zhangxt/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt", 
    # )
    args = parser.parse_args()

    if not args.fasta.exists():
        raise FileNotFoundError(args.fasta)

    args.pdb.mkdir(exist_ok=True)

    # Read fasta and sort sequences by length
    logger.info(f"Reading sequences from {args.fasta}")
    all_sequences = sorted(
        read_fasta(args.fasta), key=lambda header_seq: len(header_seq[1])
    )
    logger.info(f"Loaded {len(all_sequences)} sequences from {args.fasta}")

    logger.info("Loading model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.hub.set_dir("/pubhome/bozhang/.cache/torch/hub/")

    cfg=omegaconf.dictconfig.DictConfig( 
    content={'_name': 'ESMFoldConfig', 'esm_type': 'esm2_3B', 'fp16_esm': True, 'use_esm_attn_map': False, 'esm_ablate_pairwise': False, 'esm_ablate_sequence': False, 'esm_input_dropout': 0, 'trunk': {'_name': 'FoldingTrunkConfig', 'num_blocks': 48, 'sequence_state_dim': 1024, 'pairwise_state_dim': 128, 'sequence_head_width': 32, 'pairwise_head_width': 32, 'position_bins': 32, 'dropout': 0, 'layer_drop': 0, 'cpu_grad_checkpoint': False, 'max_recycles': 4, 'chunk_size': None, 'structure_module': {'c_s': 384, 'c_z': 128, 'c_ipa': 16, 'c_resnet': 128, 'no_heads_ipa': 12, 'no_qk_points': 4, 'no_v_points': 8, 'dropout_rate': 0.1, 'no_blocks': 8, 'no_transition_layers': 1, 'no_resnet_blocks': 2, 'no_angles': 7, 'trans_scale_factor': 10, 'epsilon': 1e-08, 'inf': 100000.0}}, 'embed_aa': True, 'bypass_lm': False, 'lddt_head_hid_dim': 128}
    )
    # model_data = torch.load(str(path_checkpoint), map_location="cpu") #读取一个pickle文件为一个dict
    # model = ESMFold(esmfold_config=cfg) # make an instance
    # model.load_state_dict(model_data['model_state_dict'], strict=False)
    # model.set_chunk_size(args.chunk_size)
    # model = model.to(device)
    # model = model.eval()
    # model.set_chunk_size(args.chunk_size)
    # print(args.parameters)
    # model = load_model_old(
    #     esm2_path="/lustre/gst/xuchunfu/zhangxt/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt",
    #     ft_path="/lustre/gst/xuchunfu/zhangxt/checkpoint/230330_tokencctop_ratio2/epoch6.pt",
    #     esmfold_path=args.parameters
    #     )
    # ft_path="/lustre/gst/xuchunfu/zhangxt/checkpoint/230330_tokencctop_ratio2/epoch6.pt"
    # esm2_path="/lustre/gst/xuchunfu/zhangxt/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt"
    # esmfold_path="/lustre/gst/xuchunfu/zhangxt/checkpoint/TPFold/ftesm2_0425_loss4/ftesm2_0425_loss4_epoch5.pt"
    ft_path="/lustre/gst/xuchunfu/zhangxt/checkpoint/230330_tokencctop_ratio2/epoch6.pt"
    esm2_path="/lustre/gst/xuchunfu/zhangxt/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt"
    esmfold_path="/lustre/gst/xuchunfu/zhangxt/checkpoint/TPFold/ftesm2_0425_loss4/ftesm2_0425_loss2_epoch13.pt"
    
    
    print(f"esmfold_path:{esmfold_path}")
    print(f"ft_path:{ft_path}")
    from TPFold import load_model_old
    model = load_model_old(
        esmfold_path=esmfold_path,
        esm2_path=esm2_path,
        ft_path=ft_path
    )
    model = model.eval()

    if args.cpu_only:
        model.cpu()
    elif args.cpu_offload:
        model = init_model_on_gpu_with_cpu_offloading(model)
    else:
        model.cuda()
    logger.info("Starting Predictions")
    batched_sequences = create_batched_sequence_datasest(
        all_sequences, args.max_tokens_per_batch
    )

    num_completed = 0
    num_sequences = len(all_sequences)
    with torch.no_grad():
        for headers, sequences in batched_sequences:
            start = timer()
            try:
                output = model(sequences, num_recycles=args.num_recycles)
            except RuntimeError as e:
                if e.args[0].startswith("CUDA out of memory"):
                    if len(sequences) > 1:
                        logger.info(
                            f"Failed (CUDA out of memory) to predict batch of size {len(sequences)}. "
                            "Try lowering `--max-tokens-per-batch`."
                        )
                    else:
                        logger.info(
                            f"Failed (CUDA out of memory) on sequence {headers[0]} of length {len(sequences[0])}."
                        )

                    continue
                raise
            del output['pred_frames']
            output["atom37_atom_exists"][..., 5:-1] = 0
            # output['positions'] = output['positions'] * mask14
            # mask14[]
            output = {key: value.cpu() for key, value in output.items()}
            pdbs = model.output_to_pdb(output)
            tottime = timer() - start
            time_string = f"{tottime / len(headers):0.1f}s"
            if len(sequences) > 1:
                time_string = time_string + f" (amortized, batch size {len(sequences)})"
            for header, seq, pdb_string, mean_plddt, ptm in zip(
                headers, sequences, pdbs, output["mean_plddt"], output["ptm"]
            ):
                output_file = args.pdb / f"{header}.pdb"
                output_file.write_text(pdb_string)
                num_completed += 1
                logger.info(
                    f"Predicted structure for {header} with length {len(seq)}, pLDDT {mean_plddt:0.1f}, "
                    f"pTM {ptm:0.3f} in {time_string}. "
                    f"{num_completed} / {num_sequences} completed."
                )
