#!/usr/bin/env python3
import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from src.tokenizer import build_bpe_tokenizer
from src.data_pipeline import make_dataloader
from src.transformer import TransformerModel

def parse_args():
    p = argparse.ArgumentParser(description="Multi‑Node DDP Transformer Training")
    p.add_argument("--train_file",    type=str, required=True,help="Path to data/wiki.train.txt")
    p.add_argument("--val_file",      type=str, required=True,help="Path to data/wiki.validation.txt")
    p.add_argument("--vocab_size",    type=int,   default=5000)
    p.add_argument("--min_frequency", type=int,   default=2)
    p.add_argument("--seq_len",       type=int,   default=256)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--emb_dim",       type=int,   default=128)
    p.add_argument("--num_heads",     type=int,   default=4)
    p.add_argument("--num_layers",    type=int,   default=3)
    p.add_argument("--num_experts",   type=int,   default=4)
    p.add_argument("--top_k_experts", type=int,   default=2)
    p.add_argument("--attn_type",     type=str,   default="grouped",choices=["multihead","grouped"])
    p.add_argument("--loss_type",     type=str,   default="multi",choices=["next","multi"])
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--epochs",        type=int,   default=10)
    return p.parse_args()

def main():
    args = parse_args()

    # These env vars are set automatically by torch.distributed.run
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize the process group
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Build tokenizer on both splits
    tokenizer = build_bpe_tokenizer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        files=[args.train_file, args.val_file]
    )

    # DataLoaders with DistributedSampler
    train_loader = make_dataloader(
        filepath=args.train_file,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=True
    )
    val_loader = make_dataloader(
        filepath=args.val_file,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=True
    )

    # Model + DDP (Distributed Data Parallel) wrapper
    model = TransformerModel(
        vocab_size=tokenizer.get_vocab_size(),
        emb_dim=args.emb_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        shared_expert_count=args.shared_expert_count,
        routed_expert_count=args.routed_expert_count,
        moe_top_k=args.moe_top_k,
        attn_type=args.attn_type,
        seq_len=args.seq_len,
        dropout=args.dropout,
        loss_type=args.loss_type
    ).to(device)
    # DDP takes your single‐GPU model and turns it into a synchronized, multi‐process model
    #   Each process (typically one per GPU) holds its own replica of the model.
    #   During the backward pass, DDP automatically averages the gradients across all replicas so that every copy stays in sync.
    #   "device_ids=[local_rank]" and "output_device=local_rank" tells DDP which GPU this process is responsible for.
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training + Validation loops
    for epoch in range(1, args.epochs + 1):
        # — Training —
        model.train()
        # Ensure each epoch shuffles differently (but ensure each process has the same shuffle)
        train_loader.sampler.set_epoch(epoch)
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            logits    = model(input_ids)
            loss      = model.module.compute_loss(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)

        # — Validation (only on rank 0) —
        if rank == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    labels    = batch["labels"].to(device)
                    logits    = model(input_ids)
                    loss      = model.module.compute_loss(logits, labels)
                    val_loss += loss.item()
            avg_val = val_loss / len(val_loader)
            print(f"[Epoch {epoch}/{args.epochs}] train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

    # Only have rank 0 save the checkpoint
    if rank == 0:
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = os.path.join("checkpoints", "transformer_poc_model.pt")
        # DDP wraps your model; use .module to get the original nn.Module
        torch.save(model.module.state_dict(), ckpt_path)
        print(f"✅ Model weights saved to {ckpt_path}")
        # save tokenizer too
        tok_path = "checkpoints/tokenizer.json"
        tokenizer.save(tok_path)  # from the tokenizers library
        print(f"✅ Tokenizer saved to {tok_path}")
    
    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
