#! /bin/bash

MASTER_NODE=$1
MASTER_PORT=$2
N_NODES=$3
shift 3

python -m torch.distributed.launch \
    --nnodes=$N_NODES --nproc_per_node=4 \
    --node_rank=${OMPI_COMM_WORLD_RANK} \
    --master_addr=$MASTER_NODE \
    --master_port=${MASTER_PORT} \
    train.py "$@"
