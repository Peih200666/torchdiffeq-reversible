import torch
import torch.nn as nn

class ReversibleODEFunc(nn.Module):
    def __init__(self, state_dim, hidden_dim_f2=32):
        super().__init__()
        if state_dim % 2 != 0:
            raise ValueError("state_dim must be even for this simple coupling layer.")
        self.dim1 = state_dim // 2
        self.dim2 = state_dim // 2

        # Mang tinh dy2/dt
        self.net2 = nn.Sequential(
            nn.Linear(self.dim1 + self.dim2 + 1, hidden_dim_f2),
            nn.Tanh(),
            nn.Linear(hidden_dim_f2, hidden_dim_f2),
            nn.Tanh(),
            nn.Linear(hidden_dim_f2, self.dim2)
        )
        self.nfe = 0

    def forward(self, t, y): # chieu thuan
        self.nfe += 1
        y1, y2 = torch.split(y, [self.dim1, self.dim2], dim=-1)

        # dy1/dt = 0
        dy1_dt = torch.zeros_like(y1)

        # dy2/dt = net2([y1, y2, t])
        if t.dim() == 0:
            t = t.reshape(1)
        t_expanded = t.expand(y.shape[0]).unsqueeze(-1).to(y.dtype)
        net2_input = torch.cat([y1, y2, t_expanded], dim=-1)
        dy2_dt = self.net2(net2_input)

        return torch.cat([dy1_dt, dy2_dt], dim=-1)

    def inverse_dynamics(self, t, y): # chieu nghich
        y1, y2 = torch.split(y, [self.dim1, self.dim2], dim=-1)
        dy1_dt_inv = torch.zeros_like(y1)

        if t.dim() == 0:
            t = t.reshape(1)
        t_expanded = t.expand(y.shape[0]).unsqueeze(-1).to(y.dtype)
        net2_input = torch.cat([y1, y2, t_expanded], dim=-1)
        dy2_dt_forward = self.net2(net2_input)
        dy2_dt_inv = -dy2_dt_forward

        return torch.cat([dy1_dt_inv, dy2_dt_inv], dim=-1)
