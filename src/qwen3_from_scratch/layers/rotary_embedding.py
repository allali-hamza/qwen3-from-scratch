import torch.nn as nn
import torch


class RotaryPositionEncoding(nn.Module):
    def __init__(self, teta_base, hidden_size, d_h):
        super().__init__()
        self.hidden_size = hidden_size
        self.teta_base = teta_base
        self.d_h = d_h

    def forward(self, x, position_ids):
        device = x.device
        num_pairs = self.d_h // 2

        tetas = torch.arange(num_pairs, device=device, dtype=torch.float32)
        tetas = 1 / (self.teta_base ** ((2 * tetas) / self.d_h))

        X_left = x[:, :, :, :self.d_h // 2]
        X_right = x[:, :, :, self.d_h // 2:]

        A = tetas * position_ids.to(
            device=device,
            dtype=torch.float32
        ).unsqueeze(-1)

        cos = torch.cos(A).to(dtype=x.dtype)
        sin = torch.sin(A).to(dtype=x.dtype)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        Y_A = X_left * cos - X_right * sin
        Y_B = X_left * sin + X_right * cos

        Y = torch.cat(
            tensors=(Y_A, Y_B),
            dim=-1
        )

        return Y

    # def forward(self, x):
    #     batch, head_size, input_lenght, d_h = x.shape
    #     num_pairs = d_h // 2

    #     device = x.device
        
    #     tetas = torch.zeros(size=(num_pairs, ), device=device)
    #     for i in range(num_pairs):
    #         teta = 1 / (self.teta_base ** ((2 * i) / d_h))
    #         tetas[i] = teta

    #     new_x = torch.zeros_like(x)
    #     for b in range(batch):
    #         for h in range(head_size):
    #             for i in range(input_lenght):
    #                 current_tetas = tetas * i
    #                 for j in range(num_pairs):
    #                     x_1 = x[b, h, i, j]
    #                     x_2 = x[b ,h ,i, num_pairs + j]

    #                     new_x[b, h, i, j] = x_1 * torch.cos(current_tetas[j]) - x_2 * torch.sin(current_tetas[j])
    #                     new_x[b, h, i, j + num_pairs] = x_1 * torch.sin(current_tetas[j]) + x_2 * torch.cos(current_tetas[j])
    #     return new_x
