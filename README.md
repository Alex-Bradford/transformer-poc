Assuming you have 2 servers each with 4 GPUs, on both machines run the below (adjust IPs, ranks, and paths accordingly)

This will spin up 8 processes (4 per node), initialize a single distributed process group across all GPUs, and run the Transformer training loop with synchronized gradients and per-epoch validation on rank 0.

--bash

export MASTER_ADDR=<ip_of_rank0_server>
export MASTER_PORT=29500

# On node 0
export NODE_RANK=0
# On node 1
# export NODE_RANK=1

# Common across both
export NNODES=2
export NPROC_PER_NODE=2

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