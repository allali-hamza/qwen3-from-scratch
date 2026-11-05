import torch

from qwen3_from_scratch.layers.rotary_embedding import RotaryPositionEncoding


def test_rotary_position_encoding_is_identity_at_position_zero():
    layer = RotaryPositionEncoding(teta_base=10_000.0, hidden_size=8, d_h=4)
    inputs = torch.randn(2, 3, 5, 4)
    position_ids = torch.zeros(2, 5, dtype=torch.long)

    outputs = layer(inputs, position_ids)

    torch.testing.assert_close(outputs, inputs)


def test_rotary_position_encoding_rotates_each_pair_by_its_position():
    layer = RotaryPositionEncoding(teta_base=10_000.0, hidden_size=2, d_h=2)
    inputs = torch.tensor([[[[2.0, 3.0], [2.0, 3.0]]]])
    position_ids = torch.tensor([[0.0, torch.pi / 2]])

    outputs = layer(inputs, position_ids)

    expected = torch.tensor([[[[2.0, 3.0], [-3.0, 2.0]]]])
    torch.testing.assert_close(outputs, expected, atol=1e-6, rtol=1e-6)


def test_rotary_position_encoding_backpropagates_to_inputs():
    layer = RotaryPositionEncoding(teta_base=10_000.0, hidden_size=8, d_h=4)
    inputs = torch.randn(2, 2, 3, 4, requires_grad=True)
    position_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])

    layer(inputs, position_ids).square().mean().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
