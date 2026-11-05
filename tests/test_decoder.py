import torch
import torch.nn as nn

from qwen3_from_scratch.layers.decoder import DecoderBlock


class ScaleAttention(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x, mask, position_ids):
        return x * self.scale


class ScaleLayer(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


def test_decoder_block_applies_both_residual_connections():
    block = DecoderBlock(hidden_size=4, h_q=2, h_kv=1, teta_base=10_000.0, d_h=2, Dff=8)
    block.first_norm = nn.Identity()
    block.attention = ScaleAttention(1.0)
    block.second_norm = nn.Identity()
    block.feed_forward = ScaleLayer(2.0)
    inputs = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    outputs = block(inputs, mask=None, position_ids=torch.tensor([[0]]))

    torch.testing.assert_close(outputs, 6 * inputs)


def test_decoder_block_preserves_shape_and_backpropagates():
    block = DecoderBlock(hidden_size=4, h_q=2, h_kv=1, teta_base=10_000.0, d_h=2, Dff=8)
    inputs = torch.randn(2, 3, 4, requires_grad=True)
    causal_mask = torch.zeros(3, 3)
    causal_mask = causal_mask.masked_fill(
        torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1),
        torch.finfo(causal_mask.dtype).min,
    ).view(1, 1, 3, 3)
    position_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])

    outputs = block(inputs, mask=causal_mask, position_ids=position_ids)
    outputs.square().mean().backward()

    assert outputs.shape == inputs.shape
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
