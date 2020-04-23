#!/usr/bin/env bash
set -u

# set environment variables for torch.distriuted
export MASTER_PORT=1234
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export RANK=$OMPI_COMM_WORLD_RANK

# Environment variables
RESULT_DIR=${RESULT_DIR:-/workspace/bert/results}
init_checkpoint=${1:-"${RESULT_DIR}/pretraining/8gpus/bert_uncased.pt"}
num_trains=${2:-3}

# parameters
precision=fp16
total_batch_size=32
learning_rate=3e-5
epochs=2.0
max_steps=-1
mode="train eval"
bert_config=$(dirname $0)/../bert/bert_config.json

batch_size=$(( $total_batch_size / $OMPI_COMM_WORLD_SIZE ))

squad_file=${DATASET_DIR}/squad.zip
vocab_file="${BERT_DATASET_DIR}/google_pretrained_weights.zip"
vocab_path="uncased_L-24_H-1024_A-16/vocab.txt"
OUT_DIR=${RESULT_DIR}/squad

CMD="${BERT_WORKSPACE_DIR}/run_squad.py "
CMD+="--init_checkpoint=$init_checkpoint "
CMD+="--squad_file=$squad_file "
CMD+=" --do_train "
CMD+=" --train_file=squad/v1.1/train-v1.1.json "
CMD+=" --train_batch_size=$batch_size "
CMD+="--do_predict "
CMD+="--predict_file=squad/v1.1/dev-v1.1.json "
CMD+="--predict_batch_size=$batch_size "
CMD+=" --do_lower_case "
CMD+=" --bert_model=bert-large-uncased "
CMD+=" --learning_rate=$learning_rate "
CMD+=" --num_train_epochs=$epochs "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --output_dir=$OUT_DIR "
CMD+=" --vocab_file=$vocab_file "
CMD+=" --vocab_path=$vocab_path "
CMD+=" --config_file=$bert_config "
CMD+=" --max_steps=-1 "
CMD+=" --fp16"
for i in $(seq $num_trains)
do
    seed=$RANDOM
    set -x
    python3 $CMD --seed $seed --local_rank ${OMPI_COMM_WORLD_LOCAL_RANK}
    set +x

    if [ "${OMPI_COMM_WORLD_RANK}" = "0" ]
    then
        python ${BERT_WORKSPACE_DIR}/../squad/v1.1/evaluate-v1.1.py ${BERT_WORKSPACE_DIR}/../squad/v1.1/dev-v1.1.json $OUT_DIR/predictions.json
        rm -rf $OUT_DIR
    fi
done
