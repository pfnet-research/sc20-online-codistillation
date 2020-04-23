#! /bin/bash

set -x

echo "Start MPI"

# Set the following parameters correctly
MASTER_NODE=master-node
MASTER_ADDR=master-addr
MASTER_PORT=1234
N_WORKERS=$1
shift 1

echo master: $MASTER_ADDR:$MASTER_NODE:$MASTER_PORT

mpiexec -N 1 -n $N_WORKERS \
    ./run.bash $MASTER_ADDR $MASTER_PORT $N_WORKERS "$@"
