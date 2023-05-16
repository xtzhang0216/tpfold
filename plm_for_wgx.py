import torch
def load_plm(esm2_path, ft_path):
        from esm.pretrained import load_regression_hub, load_model_and_alphabet_core
        model_data = torch.load(esm2_path)
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
load_plm(
    esm2_path="/pubhome/xtzhang/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt",
    ft_path="/pubhome/xtzhang/checkpoint/plm/checkpoint-6447/pytorch_model-00002-of-00002.bin")