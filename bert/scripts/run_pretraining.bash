#!/bin/bash

# set environment variables for torch.distriuted
export MASTER_PORT=1234
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export RANK=$OMPI_COMM_WORLD_RANK

# Environment variables
DATASET_PHASE1=${PHASE1_DATASET:-books_wiki_en_corpus_phase1.zip}
DATASET_PHASE2=${PHASE2_DATASET:-books_wiki_en_corpus_phase2.zip}
DATA_FILE_PHASE1=$BERT_DATASET_DIR/${DATASET_PHASE1}
DATA_FILE_PHASE2=$BERT_DATASET_DIR/${DATASET_PHASE2}
RESULTS_DIR=${RESULT_DIR}
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints

# training parameters
train_batch_size=${1:-256} # total minibatch
seed=${RANDOM}
bert_config=$(dirname $0)/../bert/bert_config.json
# LR 1e-4 / 256 batch
# warmup   10,000 (0.01)
# steps 1,000,000
total_seq=${2:-$(( 1000000 * 256 ))}
base_learning_rate=0.0001
base_warmup=0.01
ratio=$(awk "BEGIN{print $train_batch_size / 256.0}")
steps=$(awk "BEGIN{print int(${total_seq}/$train_batch_size)}")
save_checkpoint_steps=$steps # Don't save checkpoint while training
# Phase1
learning_rate=$(awk "BEGIN{print $base_learning_rate*$ratio}")
warmup_proportion=$base_warmup
train_steps_phase1=$(awk "BEGIN{print int($steps*0.9)}")
# Phase2
train_steps_phase2=$(awk "BEGIN{print int($steps*0.1)}")

CMD=" ${BERT_WORKSPACE_DIR}/run_pretraining.py"
CMD+=" --input_file=$DATA_FILE_PHASE1"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$bert_config"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_seq_length=128"
CMD+=" --max_predictions_per_seq=20"
CMD+=" --max_steps=$train_steps_phase1"
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --optimizer=adam"
CMD+=" --seed=$seed"

set -x
python3 $CMD --local_rank ${OMPI_COMM_WORLD_LOCAL_RANK}
set +x

CMD=" ${BERT_WORKSPACE_DIR}/run_pretraining.py"
CMD+=" --input_file=$DATA_FILE_PHASE2"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$bert_config"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_seq_length=512"
CMD+=" --max_predictions_per_seq=80"
CMD+=" --max_steps=$train_steps_phase2"
CMD+=" --warmup_proportion=0"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --seed=$seed"
CMD+=" --optimizer=adam"
CMD+=" --phase2 --phase1_end_step=$train_steps_phase1"

set -x
python3 $CMD --local_rank ${OMPI_COMM_WORLD_LOCAL_RANK}
set +x
