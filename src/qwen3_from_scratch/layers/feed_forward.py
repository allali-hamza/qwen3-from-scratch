import torch.nn as nn

class GatedFeedForward(nn.Module):
    def __init__(self, hidden_size, Dff):
        super().__init__()


        self.W_gate = nn.Linear(
            in_features=hidden_size,
            out_features=Dff,
            bias=False
        )

        self.W_up = nn.Linear(
            in_features=hidden_size,
            out_features=Dff,
            bias=False  
        )

        self.W_down = nn.Linear(
            in_features=Dff,
            out_features=hidden_size,
            bias=False
        )

        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.W_down(self.silu(self.W_gate(x)) * self.W_up(x))
        return x