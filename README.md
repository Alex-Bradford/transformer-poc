# transformer-poc

A lightweight proof-of-concept Transformer implementation in PyTorch, featuring:

- **Modular architecture**  
  Fully parameterized Transformer blocks with configurable depth, width, and attention style.
- **WikiText-2 placeholder**  
  Prepares local WikiText-2 splits (`data/wiki.train.txt`, `data/wiki.validation.txt`) for training.
- **Byte-level BPE tokenizer**  
  Trainable subword tokenizer with padding support.
- **Mixture-of-Experts feed-forward**  
  MoE layers in each block (shared + routed experts), inspired by DeepSeek-V2.
- **Flexible attention**  
  Switch between standard multi-head or grouped-query attention.
- **Next-token & multi-token loss**  
  Autoregressive training with 1-step or multi-step objectives.
- **Multi-node / multi-GPU training**  
  Built-in support for PyTorch DDP across servers and GPUs.

---

## 🚀 Quickstart

1. **Install dependencies**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. **Download WikiText-2 splits**
   ```bash
   python scripts/download_wikitext2.py

3. **Train on a single GPU**
   ```bash
   python scripts/train.py \
      --train_file data/wiki.train.txt \
      --val_file   data/wiki.validation.txt \
      --epochs 5 \
      --batch_size 8

---

## ⚙️ Configuration

All training scripts accept the following flags:

- **Model dimensions**: `--emb_dim`, `--num_heads`, `--num_layers`
- **MoE experts**: `--shared_expert_count`, `--routed_expert_count`, `--moe_top_k`
- **Attention type**: `--attn_type` (`multihead` \| `grouped`)
- **Loss objective**: `--loss_type` (`next` \| `multi`)
- **Other hyperparameters**: dropout, learning rate, batch size, sequence length, etc.

Run `-h` to view all options:
```bash
python scripts/train.py -h




---

## 🔗 Distributed Training

With two servers each hosting four GPUs, execute this on **both** machines (adjust IPs, ranks, and paths as needed). This will launch eight processes total (four per node), initialize a global process group, and run per-epoch validation on rank 0.

```bash
export MASTER_ADDR=<ip_of_rank0_server>
export MASTER_PORT=29500

# On node 0
export NODE_RANK=0
# On node 1
# export NODE_RANK=1

# Shared settings
export NNODES=2
export NPROC_PER_NODE=4

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/train_distributed.py \
    --train_file data/wiki.train.txt \
    --val_file   data/wiki.validation.txt \
    --epochs 10 \
    --batch_size 8

---

## 📂 Repository Layout

```bash
transformer-poc/
├── data/                    # WikiText-2 splits & other corpora
├── scripts/
│   ├── download_wikitext2.py
│   ├── train.py               # single-GPU entry point
│   └── train_distributed.py   # multi-node DDP launcher
├── src/
│   ├── tokenizer.py
│   ├── data_pipeline.py
│   ├── transformer.py
│   ├── attention.py
│   ├── ffn.py
│   └── norm.py
├── requirements.txt
└── README.md
