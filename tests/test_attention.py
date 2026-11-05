import math

import pytest
import torch

from qwen3_from_scratch.layers.attention import GroupedQueryAttention


def test_grouped_query_attention_rejects_incompatible_head_configuration():
    with pytest.raises(AssertionError):
        GroupedQueryAttention(hidden_size=6, h_q=3, h_kv=2, teta_base=10_000.0, d_h=2)


def test_attention_shares_kv_heads_and_respects_mask():
    query = torch.ones(1, 2, 2, 1)
    key = torch.tensor([[[[1.0], [2.0]]]])
    value = torch.tensor([[[[10.0], [20.0]]]])
    mask = torch.tensor([[[[0.0, torch.finfo(torch.float32).min], [0.0, 0.0]]]])

    outputs, weights = GroupedQueryAttention.attention(
        query=query, key=key, value=value, mask=mask, h_q=2, h_kv=1, d_h=1
    )

    second_token = (10.0 * math.exp(1.0) + 20.0 * math.exp(2.0)) / (math.exp(1.0) + math.exp(2.0))
    expected = torch.tensor([[[[10.0], [second_token]], [[10.0], [second_token]]]])
    torch.testing.assert_close(outputs, expected)
    torch.testing.assert_close(weights[..., 0, 1], torch.zeros_like(weights[..., 0, 1]))


def test_grouped_query_attention_forward_preserves_shape_and_backpropagates():
    layer = GroupedQueryAttention(hidden_size=4, h_q=2, h_kv=1, teta_base=10_000.0, d_h=2)
    inputs = torch.randn(2, 3, 4, requires_grad=True)
    causal_mask = torch.zeros(3, 3)
    causal_mask = causal_mask.masked_fill(
        torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1),
        torch.finfo(causal_mask.dtype).min,
    ).view(1, 1, 3, 3)
    position_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])

    outputs = layer(inputs, mask=causal_mask, position_ids=position_ids)
    outputs.square().mean().backward()

    assert outputs.shape == inputs.shape
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
