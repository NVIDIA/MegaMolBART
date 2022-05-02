#!/bin/bash
set -x

####
#Example Shell Script to run NeMo MegaMolbart data processing
#or training.  
####

### CONFIG ###
# If False, data preprocessing pipeline is run. Otherwise training runs.
MEGAMOLBART_CONFIG_FILE=megamolbart_pretrain_small_aug
DO_TRAINING="True"
DATA_MOUNT=/data/zinc_csv
CODE_MOUNT=/workspace/nemo_chem
OUTPUT_MOUNT=/result
DATA_FILES_SELECTED=x000.csv
PROJECT=MegaMolBART
DATA_FORMAT='csv'
RESULTS_MOUNT=${OUTPUT_MOUNT}/nemo_experiments/${DATA_FORMAT}/${MEGAMOLBART_CONFIG_FILE}
### END CONFIG ###

mkdir -p ${RESULTS_MOUNT}/${EXP_NAME}
export PYTHONPATH=${CODE_MOUNT}:$PYTHONPATH
export HYDRA_FULL_ERROR=1

cd ${CODE_MOUNT}/examples/chem
python megamolbart_pretrain.py \
    --config-path=conf \
    --config-name=megamolbart_pretrain_${MEGAMOLBART_CONFIG_FILE} \
    do_training=${DO_TRAINING} \
    model.data.dataset_path=${DATA_MOUNT} \
    exp_manager.exp_dir=${RESULTS_MOUNT} \
    model.data.dataset_files=${DATA_FILES_SELECTED} \
    ++tokenizer.vocab_path=${CODE_MOUNT}/nemo_chem/vocab/megamolbart.vocab \
    model.data.dataset_format=${DATA_FORMAT} 

set +x
