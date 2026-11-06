import torch
import torch.nn as nn

from qwen3_from_scratch.layers.decoder import DecoderBlock
from qwen3_from_scratch.model import TransformerBackbone


class OffsetDecoder(nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = offset
        self.received_mask = None
        self.received_position_ids = None

    def forward(self, x, mask, position_ids):
        self.received_mask = mask
        self.received_position_ids = position_ids
        return x + self.offset


def test_transformer_backbone_embeds_then_applies_each_decoder():
    first_decoder = OffsetDecoder(1.0)
    second_decoder = OffsetDecoder(2.0)
    model = TransformerBackbone(
        hidden_size=2,
        vocab_size=3,
        decoders=nn.ModuleList([first_decoder, second_decoder]),
    )
    model.norm = nn.Identity()
    with torch.no_grad():
        model.embedding.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    input_ids = torch.tensor([[0, 2]])
    mask = torch.zeros(1, 1, 2, 2)
    position_ids = torch.tensor([[4, 5]])

    outputs = model(input_ids, mask=mask, position_ids=position_ids)

    expected = torch.tensor([[[4.0, 5.0], [8.0, 9.0]]])
    torch.testing.assert_close(outputs, expected)
    assert first_decoder.received_mask is mask
    assert second_decoder.received_position_ids is position_ids


def test_transformer_backbone_preserves_sequence_shape_and_backpropagates():
    decoder = DecoderBlock(hidden_size=4, h_q=2, h_kv=1, teta_base=10_000.0, d_h=2, Dff=8)
    model = TransformerBackbone(hidden_size=4, vocab_size=10, decoders=nn.ModuleList([decoder]))
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.zeros(3, 3)
    mask = mask.masked_fill(
        torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1),
        torch.finfo(mask.dtype).min,
    ).view(1, 1, 3, 3)
    position_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])

    outputs = model(input_ids, mask=mask, position_ids=position_ids)
    outputs.square().mean().backward()

    assert outputs.shape == (2, 3, 4)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
