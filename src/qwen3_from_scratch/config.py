from dataclasses import dataclass


@dataclass
class ModelConfig():
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    def __post_init__(self):
        assert self.num_attention_heads % self.num_key_value_heads == 0

    