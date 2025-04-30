import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Rotary Position Embeddings (RoPE) -----
def apply_rope(x, freqs):
    # x: (B, T, H, D) = (Batch size =8, max seq len =256, num heads =4, head_dim =128/4=32) = (8,256,4,32)
    # freqs: (T, D/2) = (256,16)

    # move "freqs" to the same device, then add two singleton dims:
    #    -> (1, T, 1, D/2), so it can broadcast over B and H
    freqs = freqs.to(x.device).unsqueeze(0).unsqueeze(2)

    # split interleaved dims into pairs
    x1, x2 = x[..., ::2], x[..., 1::2]  # (B, T, H, D/2)

    # compute sin/cos with correct shape: both (1, T, 1, D/2)
    sin, cos = freqs.sin(), freqs.cos()

    # apply the RoPE rotation to each pair
    x_rot = torch.stack([x1 * cos - x2 * sin,
                         x1 * sin + x2 * cos], dim=-1) # (B, T, H, D/2, 2)
    
    # we flatten the last 2 dims: (B, T, H, D/2, 2) becomes (B, T, H, D)
    return x_rot.flatten(-2)

# ----- Attention Mechanisms -----
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        # more efficient to define .qkv like this, instead of 3 separate self.q = nn.Linear(emb_dim, emb_dim)
        self.qkv = nn.Linear(emb_dim, emb_dim * 3)
        self.out = nn.Linear(emb_dim, emb_dim)
    def forward(self, x, mask=None, freqs=None):
        B, T, C = x.shape
        # self.qkv(x) → shape (B, T, 3 × emb_dim)
        # .view(B, T, 3, H, D) splits the last axis into “3” (for Q/K/V), heads H, and head_dim D.
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        # .unbind(2) removes dimension 2 (the “3”), yielding three tensors: ... q:(B, T, H, D) ... k:(B, T, H, D) ... v:(B, T, H, D)
        q, k, v = qkv.unbind(2)
        # apply RoPE to query and keys, shape remains unchanged.
        if freqs is not None:
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)
        
        # k.transpose(-2,-1) swaps the last two dims of K: from (B, T, H, D) → (B, T, D, H).
        # Therefore: ... Q:(B, T, H, D) ... Kᵀ:(B, T, D, H) ... → scores: (B, T, H, H)
        # we divide by scalar (32)^0.5 for stability
        scores = (q @ k.transpose(-2, -1)) / self.head_dim**0.5
        
        # If you passed in a boolean mask of shape broadcastable to (B, T, H, H), positions where mask==0 get –∞ so softmax zeroes them.
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Applies softmax over the last axis (the “key” dimension), so for each query you get a distribution over H keys.
        # attn: shape (B, T, H, H).
        attn = F.softmax(scores, dim=-1)
        
        # Multiply attention weights by V ....  attn: (B, T, H, H) .... v: (B, T, H, D) .... → out: (B, T, H, D)
        out = attn @ v

        # Recombine the heads
        # out: (B, T, H, D)
        # .contiguous(): ensure memory is laid out so we can safely .view()
        # .view(B, T, C): merge heads back into one dimension ..... H × D = C ..... final out: (B, T, C)
        out = out.contiguous().view(B, T, C)
        
        return self.out(out)


# instead of each head having its own unique values for q, k and v ... GQA shares q across many heads (=group size)
class GroupedQueryAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, group_size=2):
        super().__init__()
        assert num_heads % group_size == 0, "num_heads must be divisible by group_size"
        self.num_heads = num_heads
        self.group_size = group_size
        self.group_count = num_heads // group_size
        self.head_dim = emb_dim // num_heads
        self.q_proj = nn.Linear(emb_dim, self.group_count * self.head_dim)
        self.kv_proj = nn.Linear(emb_dim, 2 * emb_dim)
        self.out = nn.Linear(emb_dim, emb_dim)
    def forward(self, x, mask=None, freqs=None):
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim
        G = self.group_count

        # 1) Compute grouped queries: one per group, then repeat to heads
        #    q_proj outputs (B, T, G * D)
        q = self.q_proj(x)                                              # → (B, T, G*D)
        q = q.view(B, T, G, D)                                          # → (B, T, G, D)
        q = q.repeat_interleave(self.group_size, dim=2)                # → (B, T, H, D)

        # 2) Compute full K/V projections per head
        #    kv_proj outputs (B, T, 2 * H * D)
        kv = self.kv_proj(x)                                            # → (B, T, 2*H*D)
        kv = kv.view(B, T, 2, H, D)                                     # → (B, T, 2, H, D)
        k, v = kv.unbind(2)                                             # each → (B, T, H, D)
        
        # Apply RoPE to Q and K
        if freqs is not None:
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)

        # 4) Compare query and key to calculate attention score
        #    q @ k^T over the last two dims: (B, T, H, D) @ (B, T, D, H)
        scores = (q @ k.transpose(-2, -1)) / (D)**0.5               # → (B, T, H, H)
        
        # If you passed in a boolean mask of shape broadcastable to (B, T, H, H), positions where mask==0 get –∞ so softmax zeroes them.
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 6) Softmax to get attention weights
        attn = F.softmax(scores, dim=-1)                                # → (B, T, H, H)

        # 7) Weighted sum of values
        # (B, T, H, H) @ (B, T, H, D) gives (B, T, H, D)
        out = attn @ v                                                  # → (B, T, H, D)

        # 8) Recombine heads: transpose and flatten
        out = out.contiguous().view(B, T, C)           # → (B, T, C)

        # 9) Final linear projection back to emb_dim
        return self.out(out)