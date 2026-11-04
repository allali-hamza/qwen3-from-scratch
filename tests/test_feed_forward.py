import torch
import torch.nn.functional as functional

from qwen3_from_scratch.layers.feed_forward import GatedFeedForward


def test_gated_feed_forward_preserves_leading_dimensions_and_hidden_size():
    layer = GatedFeedForward(hidden_size=4, Dff=8)
    inputs = torch.randn(2, 3, 4)

    outputs = layer(inputs)

    assert outputs.shape == (2, 3, 4)


def test_gated_feed_forward_matches_swiglu_formula():
    layer = GatedFeedForward(hidden_size=2, Dff=3)
    with torch.no_grad():
        layer.W_gate.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        layer.W_up.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]]))
        layer.W_down.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    inputs = torch.tensor([[1.0, 2.0]])

    outputs = layer(inputs)
    gate = inputs @ layer.W_gate.weight.T
    up = inputs @ layer.W_up.weight.T
    expected = (functional.silu(gate) * up) @ layer.W_down.weight.T

    torch.testing.assert_close(outputs, expected)


def test_gated_feed_forward_backpropagates_to_input_and_all_weights():
    layer = GatedFeedForward(hidden_size=3, Dff=5)
    inputs = torch.randn(2, 3, requires_grad=True)

    layer(inputs).square().mean().backward()

    assert inputs.grad is not None
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
