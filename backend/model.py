# ── model.py — Liquid Neural Network architecture (must match training exactly) 

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiquidCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_in        = nn.Linear(input_size, hidden_size)
        self.W_h         = nn.Linear(hidden_size, hidden_size)
        self.raw_tau     = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, h):
        tau = F.softplus(self.raw_tau) + 1e-3
        dh  = torch.tanh(self.W_in(x) + self.W_h(h))
        h   = h + (dh - h) / tau
        return h


class LiquidLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell        = LiquidCell(input_size, hidden_size)

    def forward(self, x, return_sequence=False, return_last=True):
        batch_size, seq_len, _ = x.size()
        h        = torch.zeros(batch_size, self.hidden_size, device=x.device)
        W_in_out = self.cell.W_in(x)
        tau      = F.softplus(self.cell.raw_tau) + 1e-3
        outputs  = []

        for t in range(seq_len):
            dh = torch.tanh(W_in_out[:, t, :] + self.cell.W_h(h))
            h  = h + (dh - h) / tau
            if return_sequence:
                outputs.append(h.unsqueeze(1))

        if return_sequence:
            out_seq = torch.cat(outputs, dim=1)
            return (out_seq, h) if return_last else out_seq
        return h


class LiquidPMModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, output_steps=24, num_targets=2):
        super().__init__()
        self.output_steps = output_steps
        self.num_targets  = num_targets
        self.liquid1      = LiquidLayer(input_size, hidden_size)
        self.liquid2      = LiquidLayer(hidden_size, hidden_size)
        self.fc           = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_steps * num_targets)
        )

    def forward(self, x):
        seq1    = self.liquid1(x, return_sequence=True, return_last=False)
        h_final = self.liquid2(seq1)
        out     = self.fc(h_final)
        return out.view(out.size(0), self.output_steps, self.num_targets)