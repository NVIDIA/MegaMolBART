#!/bin/bash
set -x

####
# Example shell script to run NeMo MegaMolbart data processing or training.  
####

### CONFIG ###

MEGAMOLBART_CONFIG_FILE=megamolbart_pretrain_xsmall_span_aug
DO_TRAINING="False" # Set to False to process data, then True to train model
DATA_FORMAT='bin' # "csv" or "bin"
DATA_MOUNT=/data/zinc_csv
DATA_FILES_SELECTED=x000.csv
CODE_MOUNT=/workspace/nemo_chem
OUTPUT_MOUNT=/result
PROJECT=MegaMolBART
RESULTS_MOUNT=${OUTPUT_MOUNT}/nemo_experiments/${DATA_FORMAT}/${MEGAMOLBART_CONFIG_FILE}

### END CONFIG ###

mkdir -p ${RESULTS_MOUNT}/${EXP_NAME}
export PYTHONPATH=${CODE_MOUNT}:$PYTHONPATH
export HYDRA_FULL_ERROR=1

cd ${CODE_MOUNT}/examples/chem
python megamolbart_pretrain.py \
    --config-path=conf \
    --config-name=${MEGAMOLBART_CONFIG_FILE} \
    do_training=${DO_TRAINING} \
    exp_manager.exp_dir=${RESULTS_MOUNT} \
    exp_manager.wandb_logger_kwargs.offline="True" \
    model.data.dataset_path=${DATA_MOUNT} \
    +model.data.dataset_format=${DATA_FORMAT} \
    model.data.dataset_files=${DATA_FILES_SELECTED}

    # tokenizer.vocab_path=${CODE_MOUNT}/nemo_chem/vocab/megamolbart.vocab \

set +x
