import torch
import torch.nn as nn

from qwen3_from_scratch.generation import generate


class Tokenizer:
    eos_token_id = 2

    def __call__(self, prompt, return_tensors):
        return {"input_ids": torch.tensor([[0]])}

    def decode(self, input_ids, skip_special_tokens):
        return str(input_ids.tolist())


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, input_ids, mask, position_ids):
        self.calls += 1
        logits = torch.zeros(1, input_ids.shape[1], 3)
        next_token = 1 if self.calls == 1 else 2
        logits[:, -1, next_token] = 1
        return logits


def test_generate_stops_when_the_model_returns_eos():
    output = generate(Model(), Tokenizer(), "prompt", max_new_tokens=10)

    assert output == "[0, 1, 2]"
