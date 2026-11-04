import torch

from qwen3_from_scratch.layers.normalization import RMSNormalization


def test_rms_normalization_matches_reference_formula():
    layer = RMSNormalization(hidden_size=3, eps=1e-6)
    inputs = torch.tensor([[3.0, 4.0, 0.0], [1.0, -1.0, 1.0]])

    outputs = layer(inputs)
    expected = inputs / torch.sqrt(inputs.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    torch.testing.assert_close(outputs, expected)


def test_rms_normalization_applies_learned_gamma():
    layer = RMSNormalization(hidden_size=2)
    with torch.no_grad():
        layer.gamma.copy_(torch.tensor([2.0, 0.5]))
    inputs = torch.tensor([[3.0, 4.0]])

    outputs = layer(inputs)
    normalized = inputs / torch.sqrt(inputs.pow(2).mean(dim=-1, keepdim=True) + layer.eps)

    torch.testing.assert_close(outputs, normalized * torch.tensor([2.0, 0.5]))


def test_rms_normalization_backpropagates_to_inputs_and_gamma():
    layer = RMSNormalization(hidden_size=3)
    inputs = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    layer(inputs).sum().backward()

    assert inputs.grad is not None
    assert layer.gamma.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(layer.gamma.grad).all()
