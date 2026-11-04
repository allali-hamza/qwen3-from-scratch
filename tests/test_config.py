import pytest


from qwen3_from_scratch.config import ModelConfig


def test_model_config_rejects_attention_heads_not_divisible_by_kv_heads():
    with pytest.raises(AssertionError):
        ModelConfig(
            vocab_size=151_936,
            hidden_size=1_024,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=3,
            head_dim=64,
            intermediate_size=3_072,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000.0,
            max_position_embeddings=32_768,
            tie_word_embeddings=False,
        )
