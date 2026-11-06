import torch
import torch.nn as nn

from qwen3_from_scratch.model import CausalLanguageModel, TransformerBackbone


def test_causal_language_model_uses_embedding_weight_as_output_head():
    backbone = TransformerBackbone(hidden_size=2, vocab_size=3, decoders=nn.ModuleList())
    backbone.norm = nn.Identity()
    with torch.no_grad():
        backbone.embedding.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    model = CausalLanguageModel(backbone)
    input_ids = torch.tensor([[0, 2]])

    logits = model(input_ids, mask=None, position_ids=torch.tensor([[0, 1]]))

    expected = torch.tensor([[[5.0, 11.0, 17.0], [17.0, 39.0, 61.0]]])
    torch.testing.assert_close(logits, expected)


def test_causal_language_model_returns_vocab_logits_and_backpropagates():
    backbone = TransformerBackbone(hidden_size=4, vocab_size=7, decoders=nn.ModuleList())
    model = CausalLanguageModel(backbone)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    position_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])

    logits = model(input_ids, mask=None, position_ids=position_ids)
    logits.square().mean().backward()

    assert logits.shape == (2, 3, 7)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
