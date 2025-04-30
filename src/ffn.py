import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Mixture of Experts -----
class MoE(nn.Module):
    def __init__(self, emb_dim, shared_expert_count, routed_expert_count, moe_top_k):
        super().__init__()
        self.shared_expert_count = shared_expert_count
        self.routed_expert_count = routed_expert_count
        self.moe_top_k = moe_top_k

        # routed experts (only top-k applied per token)
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim*4),
                nn.GELU(),
                nn.Linear(emb_dim*4, emb_dim)
            )
            for _ in range(routed_expert_count)
        ])

        # shared experts (always applied to every token)
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim*4),
                nn.GELU(),
                nn.Linear(emb_dim*4, emb_dim)
            )
            for _ in range(shared_expert_count)
        ])

        # gating only for the routed experts
        self.gate = nn.Linear(emb_dim, routed_expert_count)

    def forward(self, x):
        B, T, emb_dim = x.size()

        # Compute the shared‐expert output (average across them)
        #    shape: List of (B, T, emb_dim)  →  stack → (shared_count, B, T, emb_dim)
        shared_stack = torch.stack([expert(x) for expert in self.shared_experts], dim=0)
        # average across the shared‐expert dimension
        shared_out = shared_stack.mean(dim=0)  # → (B, T, emb_dim)

        # Compute the routed‐expert mixture
        # scores: (B, T, routed_expert_count)
        scores = F.softmax(self.gate(x), dim=-1)
        topk_vals, topk_idx = scores.topk(self.moe_top_k, dim=-1)  # (B, T, top_k)

        moe_out = torch.zeros_like(x)  # accumulator

        # for each of the top-k experts, gather and weight their outputs
        for k in range(self.moe_top_k):
            # which expert each (b,t) picks
            e_idx = topk_idx[..., k]                  # (B, T)
            coeff = topk_vals[..., k].unsqueeze(-1)   # (B, T, 1)

            # compute expert outputs for each token in the batch
            # flatten B×T → N, pick expert per token, reshape back
            flat_idx = e_idx.reshape(-1)              # (B*T,)
            flat_x   = x.reshape(-1, emb_dim)               # (B*T, emb_dim)
            # apply the corresponding expert to each flat_x[b*t + t]
            expert_out = torch.stack([
                self.routed_experts[e](flat_x[i]) for i, e in enumerate(flat_idx)
            ], dim=0).view(B, T, emb_dim)

            moe_out = moe_out + coeff * expert_out

        # Sum shared & routed
        return shared_out + moe_out