import torch 
import torch.nn as nn


class RMSNormalization(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 10**(-6)):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, x):
        input_dtype = x.dtype

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(variance + self.eps)

        x = x / rms
        x = x.to(input_dtype)

        x = x * self.gamma

        return x