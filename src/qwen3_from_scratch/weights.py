from pathlib import Path

from safetensors import safe_open


def load_pretrained_weights(model, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    weights = {}

    for file_path in sorted(checkpoint_dir.glob("*.safetensors")):
        with safe_open(file_path, framework="pt", device="cpu") as checkpoint:
            for checkpoint_name in checkpoint.keys():
                model_name = checkpoint_name

                model_name = model_name.replace("model.embed_tokens.weight", "transformer.embedding.weight")
                model_name = model_name.replace("model.norm.weight", "transformer.norm.gamma")
                model_name = model_name.replace("model.layers.", "transformer.decoders.")
                model_name = model_name.replace("input_layernorm.weight", "first_norm.gamma")
                model_name = model_name.replace("post_attention_layernorm.weight", "second_norm.gamma")
                model_name = model_name.replace("self_attn.q_proj.weight", "attention.W_q.weight")
                model_name = model_name.replace("self_attn.k_proj.weight", "attention.W_k.weight")
                model_name = model_name.replace("self_attn.v_proj.weight", "attention.W_v.weight")
                model_name = model_name.replace("self_attn.o_proj.weight", "attention.W_o.weight")
                model_name = model_name.replace("self_attn.q_norm.weight", "attention.norm_query.gamma")
                model_name = model_name.replace("self_attn.k_norm.weight", "attention.norm_key.gamma")
                model_name = model_name.replace("mlp.gate_proj.weight", "feed_forward.W_gate.weight")
                model_name = model_name.replace("mlp.up_proj.weight", "feed_forward.W_up.weight")
                model_name = model_name.replace("mlp.down_proj.weight", "feed_forward.W_down.weight")

                if checkpoint_name == "lm_head.weight":
                    model_name = "transformer.embedding.weight"

                weights[model_name] = checkpoint.get_tensor(checkpoint_name)

    model_parameters = set(model.state_dict())
    loaded_parameters = set(weights)
    missing_parameters = model_parameters - loaded_parameters
    unexpected_parameters = loaded_parameters - model_parameters

    if missing_parameters or unexpected_parameters:
        raise ValueError(
            f"Missing parameters: {sorted(missing_parameters)}\n"
            f"Unexpected parameters: {sorted(unexpected_parameters)}"
        )

    model.load_state_dict(weights, strict=True, assign=True)
