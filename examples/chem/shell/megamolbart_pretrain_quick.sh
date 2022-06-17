#!/bin/bash
####
# Example shell script to run NeMo MegaMolbart data processing or training.
####

### CONFIG ###
MEGAMOLBART_CONFIG_FILE=megamolbart_pretrain_small_span_aug
DATA_FORMAT='csv' # "csv" or "bin"
DATA_MOUNT=/data/zinc_csv
DATA_FILES_SELECTED="x[000..100].csv"
CODE_MOUNT=/workspace/nemo_chem
OUTPUT_MOUNT=/result
PROJECT=MegaMolBART
RESULTS_MOUNT=${OUTPUT_MOUNT}/nemo_experiments/${DATA_FORMAT}/${MEGAMOLBART_CONFIG_FILE}
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
        do_training=${DO_TRAINING} \
        exp_manager.exp_dir=${RESULTS_MOUNT} \
        exp_manager.wandb_logger_kwargs.offline="False" \
        model.data.dataset_path=${DATA_MOUNT} \
        +model.data.dataset_format=${DATA_FORMAT} \
        model.data.dataset.train=${DATA_FILES_SELECTED}
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
