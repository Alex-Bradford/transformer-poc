import os, sys
# assume scripts/ is one level under project root:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
from tqdm import tqdm
from src.tokenizer import build_bpe_tokenizer
from src.data_pipeline import make_dataloader
from src.transformer import TransformerModel

def parse_args():
    p = argparse.ArgumentParser(description="Single GPU Transformer Training")
    p.add_argument("--train_file",          type=str,   default="data/wiki.train.txt",help="Path to data/wiki.train.txt")
    p.add_argument("--val_file",            type=str,   default="data/wiki.validation.txt",help="Path to data/wiki.validation.txt")
    p.add_argument("--vocab_size",          type=int,   default=5000)
    p.add_argument("--min_frequency",       type=int,   default=2)
    p.add_argument("--seq_len",             type=int,   default=256)
    p.add_argument("--batch_size",          type=int,   default=8)
    p.add_argument("--num_workers",         type=int,   default=4)
    p.add_argument("--emb_dim",             type=int,   default=128)
    p.add_argument("--num_blocks",          type=int,   default=2)
    p.add_argument("--num_heads",           type=int,   default=4)
    p.add_argument("--shared_expert_count", type=int,   default=1)
    p.add_argument("--routed_expert_count", type=int,   default=2)
    p.add_argument("--moe_top_k",           type=int,   default=1)
    p.add_argument("--attn_type",           type=str,   default="grouped",choices=["multihead","grouped"])
    p.add_argument("--loss_type",           type=str,   default="multi",choices=["next","multi"])
    p.add_argument("--dropout",             type=float, default=0.1)
    p.add_argument("--lr",                  type=float, default=1e-4)
    p.add_argument("--epochs",              type=int,   default=1)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build BPE tokenizer on train+val
    tokenizer = build_bpe_tokenizer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        files=[args.train_file, args.val_file]
    )

    # Create DataLoaders (non‑distributed)
    train_loader = make_dataloader(
        filepath=args.train_file,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=False
    )
    val_loader = make_dataloader(
        filepath=args.val_file,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=False
    )

    # Instantiate model
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training + Validation loop
    for epoch in range(1, args.epochs + 1):
        # we put the model in training mode, which turns dropout layers on (useful for generalisation)
        model.train()
        # we reset the training loss to 0 for each epoch (not each batch)
        train_loss = 0.0
        
        # loop each batch
        # for batch in train_loader:
        # use tqdm for progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False, unit="batch"):
            # we need to reset the gradients in the optimiser for every batch, otherwise the gradients will keep accumulating and provide incorrect values (thus won’t minimise loss). 
            optimizer.zero_grad()
            
            # by default, DataLoader returns tensors on the CPU, 
            # if we're running the model on a GPU this will cause an error when we use model.compute_loss(logits, labels), because everything needs to be on the same device.
            # Therefore, we move inputs and labels onto the same device as the model
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            
            # run a forward pass (noting again, we input logits into the cross entropy loss function)
            logits    = model(input_ids)
            loss      = model.compute_loss(logits, labels)
            
            # this line computes gradients of loss w.r.t every param, and populates each params .grad attribute.
            loss.backward()
            
            # This line reads each param's .grad attribute and applies the optimisation rule
            optimizer.step()
            
            # capture the value of the loss fn from this batch
            train_loss = train_loss + loss.item()
        avg_train = train_loss / len(train_loader)

        # we put the model in eval mode, which turns dropout layers OFF (we don't want to randomly 0 out some input values when evaluation the model)
        model.eval()
        # we reset the val loss to 0 for each epoch (not each batch)
        val_loss = 0.0

        # we disable grading tracking for the validation data to save memory and speed up computation 
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels    = batch["labels"].to(device)
                logits    = model(input_ids)
                loss      = model.compute_loss(logits, labels)
                val_loss = val_loss + loss.item()
        avg_val = val_loss / len(val_loader)

        print(
            f"[Epoch {epoch}/{args.epochs}]  "
            f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}"
        )
    
    # save the model
    # Only have rank 0 save the checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "transformer_poc_model.pt")
    # DDP wraps your model; use .module to get the original nn.Module
    torch.save(model.state_dict(), ckpt_path)
    print(f"✅ Model weights saved to {ckpt_path}")
    # save tokenizer too
    tok_path = "checkpoints/tokenizer.json"
    tokenizer.save(tok_path)  # from the tokenizers library
    print(f"✅ Tokenizer saved to {tok_path}")

if __name__ == "__main__":
    main()
