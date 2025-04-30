import torch
import torch.nn as nn
import torch.nn.functional as F

from src.norm import RMSNorm
from src.attention import MultiHeadAttention, GroupedQueryAttention
from src.ffn import MoE

# ----- Transformer Block -----
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, shared_expert_count, routed_expert_count, moe_top_k, dropout, 
                 attn_type='multihead', group_size=2):
        super().__init__()
        self.rms1 = RMSNorm(emb_dim)
        
        # select attention
        if attn_type == 'multihead':
            self.attn = MultiHeadAttention(emb_dim, num_heads)
        elif attn_type == 'grouped':
            self.attn = GroupedQueryAttention(emb_dim, num_heads, group_size)
        else:
            raise ValueError(f"Unknown attn_type {attn_type}")
        
        # Dropout after attention
        self.attn_dropout = nn.Dropout(dropout)
        
        # norm again
        self.rms2 = RMSNorm(emb_dim)
        
        # MoE / FFN layer
        self.moe = MoE(emb_dim, shared_expert_count, routed_expert_count, moe_top_k)

        # Dropout after MoE/FFN
        self.moe_dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None, freqs=None):
        # Pre-attention normalization
        x_norm = self.rms1(x)

        # Attention + dropout
        attn_out = self.attn(x_norm, mask=mask, freqs=freqs)
        attn_out = self.attn_dropout(attn_out)

        # residual connection
        x = x + attn_out

        # Pre-FFN/MoE normalization
        x_norm = self.rms2(x)
        # MoE/FFN + dropout
        moe_out = self.moe(x_norm)
        moe_out = self.moe_dropout(moe_out)
        # Final residual
        return x + moe_out

# ----- Transformer Model -----
'''
inherit attributes and methods from nn.Module (this is the base class for ALL NN components in PyTorch), eg. useful ones:
.parameters() yields all weights for the optimizer
.to(device) moves every sub-module to the GPU
.train() and .eval() to toggle modes

use super().__init__() to run nn.Module.__init__()

When you define any part of the Neural Network, you should also include the [nn.Module] and [super().__init__()] boilerplate
'''
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_blocks, num_heads,
                 shared_expert_count, routed_expert_count, moe_top_k, attn_type, seq_len, dropout,
                 loss_type='next', group_size=2):
        super().__init__()
        
        # Create the learnable lookup table which maps (vocab_size) to (emb_dim)
        # the input token IDs will have shape (B, T) with B = batch size and T = seq len
        # token_emb(input_ids) will return (B, T, emb_dim)
        # it's common practice to set emb_dim so it is divisible by num_heads
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        
        # we will split the emb_dim into equal heads. Each head projects into head_dim 
        head_dim = emb_dim // num_heads

        # torch.arange(0, head_dim, 2) gives indices [0,2,4,…] up to head_dim. Then divide by head_dim so we get [0, ..., 1]
        # inv_freq = [1,... 0.1, .... 0.001, ..., 0.0001 ] with len = head_dim/2
        # in RoPE we treat dimensions in pairs (d, d+1) so that we can apply the 2x2 rotation matrix. Therefore we only need one frequency per PAIR of dimensions. 
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        
        # create a list of indices up to seq_len eg. [0, 1, ..., seq_len -1]
        t = torch.arange(seq_len, dtype=torch.float)

        # freqs is a matrix of shape (seq_len, head_dim/2)
        # [0*1=0    ...    0*0.001=0    ...     0*0.0001=0    ]
        # ..
        # [128*1=128  ... 128*0.001=0.128 ... 128*0.0001=0.0128]
        # ..
        # [255*1=255  ... 255*0.001=0.255 ... 255*0.0001=0.0255]
        # we will later sin/cos these frequencies to rotate the queries/keys in each block
        self.freqs = torch.einsum('i,j->ij', t, inv_freq)
        
        # use nn.ModuleList() here, otherwise A plain Python list of TransformerBlock objects would not be picked up by nn.Module’s machinery.
        # ModuleList tells PyTorch: “Here are num_blocks child modules, register them so their parameters get tracked."
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, shared_expert_count, routed_expert_count, moe_top_k, dropout, attn_type, group_size) for _ in range(num_blocks)
            ])
        
        # define the final Norm layer and then the head to project emd_dim back to vocab.
        self.rms_final = RMSNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)
        self.loss_type = loss_type

    def forward(self, input_ids, mask=None):
        B, T = input_ids.shape
        x = self.token_emb(input_ids)
        freqs = self.freqs[:T].to(x.device)
        for block in self.blocks:
            x = block(x, mask=mask, freqs=freqs)
        x = self.rms_final(x)
        return self.head(x)

    def compute_loss(self, logits, labels):
        # logits are the raw output of the network, shape is (Batch, Seq_Len, Vocab). 
        # -- Don't use softmax (cross_entropy Fn will do that internally)
        # labels are the correct token IDs, shape is (Batch, Seq_Len), with the values being the token ID
        # we want to predict token t given all tokens up to t-1

        # before we input the prediction and target vectors into cross-entropy; 
        # -- we need to remove the last prediction (there is no label to compare it to)
        # -- we need to remove the first label (there is no prediction to compare it to), and also implicitly shift the labels back 1 index to align them with their corresponding predictions
        # -- Then we end up with logits=(Batch, Seq_Len - 1, Vocab) AND labels=(Batch, Seq_Len - 1)
        # we use .contiguous() to make sure the tensor is laid out in memory without any weird strides, which is important before you call .view()

        # torch.Tensor.view() is used to reshape a tensor
        # using the value "-1" means to infer the shape from other dimensions.
        # -- because we used "shifted_labels.view(-1)" this means to flatten the whole thing
        # -- because we used "logits.view(   -1,   logits.size(-1)   )", this means reshape to (x, Vocab size), where x will be inferred from the other dimensions and be made to fit with 2nd_dim=Vocab_size

        if self.loss_type == 'next':
            logits = logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                shifted_labels.view(-1), 
                ignore_index=-100)

        if self.loss_type == 'multi':
            # Step k = 1 is exactly your existing “next‐token” loss: it matches logits[b, t] to labels[b, t+1] for all t = 0…T-2.
            # Step k = 2 matches logits[b, t] to labels[b, t+2] for t = 0…T-3.
            # Step k = 3 matches logits[b, t] to labels[b, t+3] for t = 0…T-4.
            
            steps = [1, 2, 3]   # how many tokens ahead to predict
            losses = []
            V = logits.size(-1)

            for k in steps:
                # 1) chop off the last k logits (they have no corresponding label)
                shifted_logits = logits[..., :-k, :].contiguous()   # [B, T-k, V]
                # 2) chop off the first k labels (they have no corresponding logit)
                shifted_labels = labels[..., k:].contiguous()       # [B, T-k]

                # 3) flatten and compute CE
                loss_k = F.cross_entropy(
                    shifted_logits.view(-1, V),      # [B*(T-k), V]
                    shifted_labels.view(-1),         # [B*(T-k)]
                    ignore_index=-100
                )
                losses.append(loss_k)

            # 4) average the three losses
            return torch.stack(losses, dim=0).mean()