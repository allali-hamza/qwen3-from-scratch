import torch
import torch.nn as nn
from math import sqrt

from qwen3_from_scratch.layers.normalization import RMSNormalization
from qwen3_from_scratch.layers.rotary_embedding import RotaryPositionEncoding


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, h_q, h_kv, teta_base, d_h):
        super().__init__()

        self.hidden_size = hidden_size
        self.h_q = h_q
        self.h_kv = h_kv
        self.d_h = d_h
        self.teta_base = teta_base

        assert self.hidden_size == self.h_q * self.d_h
        assert self.h_q % self.h_kv == 0, "h_q should be divisible by h_kv"

        self.W_q = nn.Linear(
            in_features=hidden_size,
            out_features=self.d_h * self.h_q,
            bias=False
        )

        self.W_k = nn.Linear(
            in_features=hidden_size,
            out_features=self.d_h * self.h_kv,
            bias=False
        )

        self.W_v = nn.Linear(
            in_features=hidden_size,
            out_features=self.d_h * self.h_kv,
            bias=False
        )

        self.W_o = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False
        )

        self.norm_query = RMSNormalization(
            hidden_size=self.d_h
        )

        self.norm_key = RMSNormalization(
            hidden_size=self.d_h
        )

        self.rot_embedding = RotaryPositionEncoding(
            teta_base=teta_base,
            hidden_size=self.hidden_size,
            d_h=self.d_h
        )

    @staticmethod
    def attention(query, key, value, mask, h_q, h_kv, d_h):

        G = h_q // h_kv

        # Q: (B, Hq, T, Dh) -> (B, Hkv, G, T, Dh)
        query = query.reshape(
            query.shape[0],
            h_kv,
            G,
            query.shape[2],
            query.shape[3]
        )

        # K: (B, Hkv, T, Dh) -> (B, Hkv, 1, Dh, T)
        key = key.unsqueeze(2).transpose(-2, -1)

        # V: (B, Hkv, T, Dh) -> (B, Hkv, 1, T, Dh)
        value = value.unsqueeze(2)

        # Attention scores: (B, Hkv, G, Tq, Tk)
        scores = (query @ key) * (1 / sqrt(d_h))

        if mask is not None:
            if mask.ndim == 4:
                mask = mask.unsqueeze(2)

            scores = scores + mask

        
        attention_weights = torch.softmax(
            scores,
            dim=-1,
            dtype=torch.float32
        ).to(query.dtype)

        output = attention_weights @ value

        # (B, Hkv, G, Tq, Dh) -> (B, Hq, Tq, Dh)
        output = output.reshape(
            output.shape[0],
            h_q,
            output.shape[3],
            output.shape[4]
        )

        return output, attention_weights

    def forward(self, x, mask, position_ids):

        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)

        query = query.reshape(
            query.shape[0],
            query.shape[1],
            self.h_q,
            self.d_h
        )
        query = query.transpose(1, 2)

        key = key.reshape(
            key.shape[0],
            key.shape[1],
            self.h_kv,
            self.d_h
        )
        key = key.transpose(1, 2)

        value = value.reshape(
            value.shape[0],
            value.shape[1],
            self.h_kv,
            self.d_h
        )
        value = value.transpose(1, 2)

        query = self.norm_query(query)
        key = self.norm_key(key)

        query = self.rot_embedding(query, position_ids)
        key = self.rot_embedding(key, position_ids)

        results, _ = GroupedQueryAttention.attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            h_q=self.h_q,
            h_kv=self.h_kv,
            d_h=self.d_h
        )

        results = results.transpose(1, 2)
        results = results.reshape(
            results.shape[0],
            results.shape[1],
            self.hidden_size
        )

        output = self.W_o(results)

        return output