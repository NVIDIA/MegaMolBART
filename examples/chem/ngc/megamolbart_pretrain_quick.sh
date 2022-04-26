#!/bin/bash
set -x

##### Tested with single node, multiple GPU configuration
### CONFIG ###

# If False, data preprocessing pipeline is run. Otherwise training runs.
DO_TRAINING="True"


DATA_MOUNT=/data/zinc_csv
CODE_MOUNT=/workspace/nemo_chem
OUTPUT_MOUNT=/result

JOB_NUM_NODES=1
GPUS_PER_NODE=1

MEGAMOLBART_CONFIG_FILE=small_aug
DATA_FILES_SELECTED=x000.csv

HOSTNAME=NGC
PROJECT=MegaMolBART
ENCODER_TYPE=${1:-seq2seq}
### END CONFIG ###

EXP_NAME=${HOSTNAME}_nodes_${JOB_NUM_NODES}_gpus_${GPUS_PER_NODE}
RESULTS_MOUNT=${OUTPUT_MOUNT}/nemo_experiments/${PROJECT}/csv1/${MEGAMOLBART_CONFIG_FILE}

mkdir -p ${RESULTS_MOUNT}/${EXP_NAME}
GPU_LIMIT=$(($GPUS_PER_NODE-1))
SCRIPT_CUDA_VISIBLE_DEVICES=$(seq --separator=',' 0 $GPU_LIMIT)

export CUDA_VISIBLE_DEVICES=${SCRIPT_CUDA_VISIBLE_DEVICES}
export PYTHONPATH=${CODE_MOUNT}:$PYTHONPATH
export HYDRA_FULL_ERROR=1

cd ${CODE_MOUNT}/examples/chem
if [ -z ${WANDB_API_KEY} ]; then
    WANDB_API_KEY=$(grep password $HOME/.netrc | cut -d' ' -f4)
fi

if [ -z ${WANDB_API_KEY} ]; then 
    WANDB_OFFLINE_MODE="true" # handle api key failures gracefully
else
    WANDB_OFFLINE_MODE="false"
fi

python megamolbart_pretrain.py \
    --config-path=conf \
    --config-name=megamolbart_pretrain_${MEGAMOLBART_CONFIG_FILE} \
    do_training=${DO_TRAINING} \
    model.data.dataset_path=${DATA_MOUNT} \
    exp_manager.wandb_logger_kwargs.offline=${WANDB_OFFLINE_MODE} \
    exp_manager.wandb_logger_kwargs.job_type=${EXP_NAME} \
    exp_manager.name=${EXP_NAME} \
    exp_manager.exp_dir=${RESULTS_MOUNT} \
    trainer.num_nodes=${JOB_NUM_NODES} \
    trainer.devices=${GPUS_PER_NODE} \
    model.data.dataset_files=${DATA_FILES_SELECTED} \
    model.data.num_workers=4 \
    ++tokenizer.vocab_path=${CODE_MOUNT}/nemo_chem/vocab/megamolbart.vocab \
    trainer.val_check_interval=0.5 \
    ++trainer.limit_train_batches=10 \
    ++trainer.max_epochs=10 \
    model.data.dataset_format='csv' 

set +x
