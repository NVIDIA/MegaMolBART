#!/bin/bash
set -x

##### Interactive training / development on a cluster with SLURM
### CONFIG ###

NUM_GPUS=SLURM_GPUS_PER_NODE
NUM_NODES=SLURM_JOB_NUM_NODES

PROJECT=MegaMolBART
MEGAMOLBART_CONFIG_FILE=megamolbart_pretrain_small_span_aug
DATA_FILES_SELECTED=x_OP_000..001_CL_.csv
CONTAINER="nvcr.io#nvidian/clara-lifesciences/megamolbart_training_nemo:210828"
WANDB_API_KEY=$(grep password $HOME/.netrc | cut -d' ' -f4)
STORAGE_DIR=/gpfs/fs1/projects/ent_joc/users/mgill/megatron

DATA_DIR=${STORAGE_DIR}/data/zinc_csv_split
CODE_DIR=${STORAGE_DIR}/code/NeMo
OUTPUT_DIR=${STORAGE_DIR}/nemo
RESULTS_DIR=${OUTPUT_DIR}/${EXP_NAME}
OUTFILE="${RESULTS_DIR}/slurm-%j-%n.out" # Not created in interactive mode
ERRFILE="${RESULTS_DIR}/error-%j-%n.out" # Not created in interactive mode

### END CONFIG ###

HOSTNAME=$(hostname)
HOSTNAME=${HOSTNAME%%"-login"*} # remove login string from name
EXP_NAME=${HOSTNAME}_nodes_${NUM_NODES}_gpus_${NUM_GPUS}

NTASKS=$((${NUM_NODES}*${NUM_GPUS}))
DATA_MOUNT=/data
CODE_MOUNT=/code
OUTPUT_MOUNT=/result
RESULTS_MOUNT=${OUTPUT_MOUNT}/${EXP_NAME}
MOUNTS="$CODE_DIR:$CODE_MOUNT,$OUTPUT_DIR:$OUTPUT_MOUNT,$DATA_DIR:$DATA_MOUNT"
WORKDIR=${CODE_MOUNT}

mkdir -p ${RESULTS_DIR}
GPU_LIMIT="$(($NUM_GPUS-1))"
SCRIPT_CUDA_VISIBLE_DEVICES=$(seq --separator=',' 0 $GPU_LIMIT)
SCRIPT_PYTHONPATH=${CODE_MOUNT}':$PYTHONPATH'

if [ -z ${WANDB_API_KEY} ]; then
    WANDB_API_KEY=$(grep password $HOME/.netrc | cut -d' ' -f4)
fi

if [ -z ${WANDB_API_KEY} ]; then 
    WANDB_OFFLINE_MODE="true" # handle api key failures gracefully
else
    WANDB_OFFLINE_MODE="false"
fi

read -r -d '' RUN_COMMAND << EOF
echo '*******STARTING********' \
&& echo '---------------' \
&& cd ${CODE_MOUNT}/examples/chem \
&& echo 'Starting training' \
&& export CUDA_VISIBLE_DEVICES=${SCRIPT_CUDA_VISIBLE_DEVICES} \
&& export PYTHONPATH=${SCRIPT_PYTHONPATH} \
&& export HYDRA_FULL_ERROR=1 \
&& python megamolbart_pretrain.py \
    --config-path=conf \
    --config-name=${MEGAMOLBART_CONFIG_FILE} \
    exp_manager.wandb_logger_kwargs.offline=${WANDB_OFFLINE_MODE} \
    exp_manager.wandb_logger_kwargs.job_type=${EXP_NAME} \
    exp_manager.name=${EXP_NAME} \
    exp_manager.exp_dir=${RESULTS_MOUNT} \
    trainer.num_nodes=${NUM_NODES} \
    trainer.gpus=${NUM_GPUS} \
    tokenizer.vocab_path=${CODE_MOUNT}/nemo/collections/chem/vocab/megamolbart_pretrain_vocab.txt \
    model.train_ds.filepath=${DATA_MOUNT}/train/${DATA_FILES_SELECTED} \
    model.train_ds.metadata_path=${DATA_MOUNT}/train/metadata.txt \
    model.train_ds.num_workers=10 \
    model.validation_ds.filepath=${DATA_MOUNT}/val/${DATA_FILES_SELECTED} \
    model.validation_ds.metadata_path=${DATA_MOUNT}/val/metadata.txt \
    model.validation_ds.num_workers=4
EOF


SCRIPT_PATH=${RESULTS_DIR}/job_script.sh
echo "${RUN_COMMAND}" > ${SCRIPT_PATH}
export SCRIPT_MOUNT=${RESULTS_MOUNT}/job_script.sh

srun --pty \
--gres=gpfs:circe \
--account ent_joc_model_mpnn_pyt \
--partition interactive \
--nodes ${NUM_NODES} \
--ntasks ${NTASKS} \
--ntasks-per-node ${NUM_GPUS} \
--gpus-per-node ${NUM_GPUS} \
--container-image ${CONTAINER} \
--container-mounts ${MOUNTS} \
--container-workdir ${WORKDIR} \
--export PYTHONPATH="${SCRIPT_PYTHONPATH}" \
--export RUN_COMMAND="${RUN_COMMAND}" \
--export SCRIPT_PATH="${SCRIPT_MOUNT}" \
--nv-meta ml-model.megamolbart_int \
bash

# bash ${OUTPUT_MOUNT}/${EXP_NAME}/job_script.sh 
# bash -c "${RUN_COMMAND}"

set +x
