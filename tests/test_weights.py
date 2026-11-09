import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from qwen3_from_scratch.layers.decoder import DecoderBlock
from qwen3_from_scratch.model import CausalLanguageModel, TransformerBackbone
from qwen3_from_scratch.weights import load_pretrained_weights


def make_model():
    decoder = DecoderBlock(hidden_size=4, h_q=2, h_kv=1, teta_base=10_000.0, d_h=2, Dff=6)
    backbone = TransformerBackbone(hidden_size=4, vocab_size=7, decoders=nn.ModuleList([decoder]))
    return CausalLanguageModel(backbone)


def checkpoint_name(model_name):
    checkpoint_name = model_name
    checkpoint_name = checkpoint_name.replace("transformer.embedding.weight", "model.embed_tokens.weight")
    checkpoint_name = checkpoint_name.replace("transformer.norm.gamma", "model.norm.weight")
    checkpoint_name = checkpoint_name.replace("transformer.decoders.", "model.layers.")
    checkpoint_name = checkpoint_name.replace("first_norm.gamma", "input_layernorm.weight")
    checkpoint_name = checkpoint_name.replace("second_norm.gamma", "post_attention_layernorm.weight")
    checkpoint_name = checkpoint_name.replace("attention.W_q.weight", "self_attn.q_proj.weight")
    checkpoint_name = checkpoint_name.replace("attention.W_k.weight", "self_attn.k_proj.weight")
    checkpoint_name = checkpoint_name.replace("attention.W_v.weight", "self_attn.v_proj.weight")
    checkpoint_name = checkpoint_name.replace("attention.W_o.weight", "self_attn.o_proj.weight")
    checkpoint_name = checkpoint_name.replace("attention.norm_query.gamma", "self_attn.q_norm.weight")
    checkpoint_name = checkpoint_name.replace("attention.norm_key.gamma", "self_attn.k_norm.weight")
    checkpoint_name = checkpoint_name.replace("feed_forward.W_gate.weight", "mlp.gate_proj.weight")
    checkpoint_name = checkpoint_name.replace("feed_forward.W_up.weight", "mlp.up_proj.weight")
    checkpoint_name = checkpoint_name.replace("feed_forward.W_down.weight", "mlp.down_proj.weight")
    return checkpoint_name


def write_checkpoint(checkpoint_dir, model):
    weights = {}
    expected = {}
    for index, (model_name, tensor) in enumerate(model.state_dict().items(), start=1):
        tensor = torch.full_like(tensor, float(index))
        weights[checkpoint_name(model_name)] = tensor
        expected[model_name] = tensor

    save_file(weights, checkpoint_dir / "model.safetensors")
    return expected


def test_load_pretrained_weights_remaps_and_loads_all_parameters(tmp_path):
    model = make_model()
    expected = write_checkpoint(tmp_path, model)

    load_pretrained_weights(model, tmp_path)

    for name, tensor in expected.items():
        torch.testing.assert_close(model.state_dict()[name], tensor)


def test_load_pretrained_weights_rejects_missing_parameters(tmp_path):
    model = make_model()
    expected = write_checkpoint(tmp_path, model)
    missing_model_name = next(iter(expected))
    checkpoint_path = tmp_path / "model.safetensors"

    weights = {
        checkpoint_name(model_name): tensor
        for model_name, tensor in expected.items()
        if model_name != missing_model_name
    }
    save_file(weights, checkpoint_path)

    with pytest.raises(ValueError, match="Missing parameters"):
        load_pretrained_weights(model, tmp_path)
