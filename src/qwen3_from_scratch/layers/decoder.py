import torch
import torch.nn as nn

from qwen3_from_scratch.layers.normalization import RMSNormalization
from qwen3_from_scratch.layers.attention import GroupedQueryAttention
from qwen3_from_scratch.layers.feed_forward import GatedFeedForward

class DecoderBlock(nn.Module):
    def __init__(self, hidden_size, h_q, h_kv, teta_base, d_h, Dff):
        super().__init__()

        self.first_norm = RMSNormalization(
            hidden_size=hidden_size
        )

        self.second_norm = RMSNormalization(
            hidden_size=hidden_size
        )

        self.attention = GroupedQueryAttention(
            hidden_size=hidden_size,
            h_q=h_q,
            h_kv=h_kv,
            teta_base=teta_base,
            d_h=d_h
        )

        self.feed_forward = GatedFeedForward(
            hidden_size=hidden_size,
            Dff=Dff
        )

    def forward(self, x, mask, position_ids):
        residual = x
        x = self.first_norm(x)
        x = self.attention(x, mask, position_ids)
        H = x + residual

        stored_H = H

        x = self.second_norm(H)
        x = self.feed_forward(x)
        result = x + stored_H

        return result
        
