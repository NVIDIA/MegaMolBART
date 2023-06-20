#!/bin/bash
####
# Example shell script to run NeMo MegaMolbart data processing or training.
####

### CONFIG ###
MEGAMOLBART_CONFIG_FILE=megamolbart_pretrain_xsmall_span_aug
DATA_FORMAT='csv' # "csv" or "bin"
DATA_MOUNT=/data/zinc_csv_split
CODE_MOUNT=/opt/nvidia/nemo_chem
OUTPUT_MOUNT=/result
PROJECT=MegaMolBART
RESULTS_MOUNT=${OUTPUT_MOUNT}/nemo_experiments/${DATA_FORMAT}/${MEGAMOLBART_CONFIG_FILE}
DATA_FILES_SELECTED=x_OP_000..001_CL_ #x000
WANDB_OFFLINE=True

TRAINING_ARGS="exp_manager.exp_dir=${RESULTS_MOUNT}"
TRAINING_ARGS="${TRAINING_ARGS} exp_manager.wandb_logger_kwargs.offline=${WANDB_OFFLINE}"
TRAINING_ARGS="${TRAINING_ARGS} model.data.dataset_path=${DATA_MOUNT}"
TRAINING_ARGS="${TRAINING_ARGS} model.data.dataset_format=${DATA_FORMAT}"
### END CONFIG ###


usage() {
cat <<EOF
USAGE: megamolbart_pretrain_quick.sh
megamolbart pretrain script
----------------------------------------
megamolbart_pretrain_quick.sh [command]
    valid commands:
        preprocess
        train

    default command:
        train

    options:
        -f|--data-files
            List of data files to use
        --data-format
            Training data format. Valid values: "csv" or "bin".
        -c|--config
            Configuration

EOF
}


execute() {
    set -x
    python megamolbart_pretrain.py \
        --config-path=conf \
        --config-name=${MEGAMOLBART_CONFIG_FILE} \
        do_training=${DO_TRAINING}
        ${TRAINING_ARGS}
    set +x
}


preprocess() {
    DO_TRAINING="False"
    parse_args $@
    execute
}


train() {
    DO_TRAINING="True"
    parse_args $@
    execute
}


parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--data-files)
                DATA_FILES_SELECTED="$2"
                shift
                shift
                ;;
            --data-format)
                DATA_FORMAT="$2"
                shift
                shift
                ;;
            -c|--config)
                MEGAMOLBART_CONFIG_FILE="$2"
                shift
                shift
                ;;
            -a|--args)
                TRAINING_ARGS="${TRAINING_ARGS} $2"
                shift
                shift
                ;;
            *)
                usage
                exit 1
                ;;
        esac
    done
}


mkdir -p ${RESULTS_MOUNT}/${EXP_NAME}
export PYTHONPATH=${CODE_MOUNT}:$PYTHONPATH
export HYDRA_FULL_ERROR=1
cd ${CODE_MOUNT}/examples/chem

if [ $# -eq 0 ]; then
    ARGS=train
    CMD='train'
else
    ARGS=$1
    CMD=$@
fi

case $ARGS in
    preprocess)
        $CMD
        ;;
    train)
        $CMD
        ;;
    *)
        usage
        exit 1
        ;;
esac
