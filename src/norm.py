import torch
import torch.nn as nn

# ----- RMSNorm -----
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        # create the learnable scale vector
        self.weight = nn.Parameter(torch.ones(emb_dim))
    def forward(self, x):
        # x.pow(2).mean(01, keepdim=True) ---> compute mean of squares over features → E[x²]
        # .rsqrt() .....  → 1/√(E[x²] + ε)
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight