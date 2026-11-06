import torch 
import torch.nn as nn
from qwen3_from_scratch.layers.decoder import DecoderBlock
from qwen3_from_scratch.layers.normalization import RMSNormalization

class TransformerBackbone(nn.Module):
    def __init__(self, hidden_size, vocab_size, decoders: nn.ModuleList):
        super().__init__()

        self.decoders = decoders

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )

        self.norm = RMSNormalization(
            hidden_size=hidden_size
        )

    def forward(self, x, mask, position_ids):
        x = self.embedding(x)
        for decoder in self.decoders:
            x = decoder(x, mask, position_ids)

        x = self.norm(x)

        return x


class CausalLanguageModel(nn.Module):
    def __init__(self, transformer: TransformerBackbone):
        super().__init__()
        self.transformer = transformer

    def forward(self, x, mask, position_ids):
        logits = self.transformer(x, mask, position_ids) @ self.transformer.embedding.weight.T
        return logits