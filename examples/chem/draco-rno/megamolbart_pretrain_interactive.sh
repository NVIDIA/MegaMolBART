#!/bin/bash
set -x

##### Interactive training / development on a cluster with SLURM
# Tested with single node, multiple GPU configuration

### CONFIG ###
SLURM_JOB_NUM_NODES=1
SLURM_GPUS_PER_NODE=2

MEGAMOLBART_CONFIG_FILE=megamolbart_pretrain_small_span_aug
DATA_FILES_SELECTED=x_OP_000..001_CL_.csv
CONTAINER="nvcr.io#nvidian/clara-lifesciences/megamolbart_training_nemo:210716"
WANDB=88800d16aea5891a1cdab809b2c47c351c8125e1
STORAGE_DIR=/gpfs/fs1/projects/ent_joc/users/mgill/megatron

PROJECT=MegaMolBART # exp_manager and wandb
EXPNAME=Draco-RNO # exp_manager and wandb
EXP_DIR=${EXPNAME}_nodes_${SLURM_JOB_NUM_NODES}_gpus_${SLURM_GPUS_PER_NODE}

DATA_DIR=${STORAGE_DIR}/data/zinc_csv_split
CODE_DIR=${STORAGE_DIR}/code/NeMo
OUTPUT_DIR=${STORAGE_DIR}/nemo

### 
NTASKS=$((${SLURM_JOB_NUM_NODES}*${SLURM_GPUS_PER_NODE}))
RESULTS_DIR=${OUTPUT_DIR}/${EXP_DIR}
mkdir -p ${RESULTS_DIR}

DATA_MOUNT=/data
CODE_MOUNT=/code
OUTPUT_MOUNT=/result
RESULTS_MOUNT=${OUTPUT_MOUNT}/${EXP_DIR}
WORKDIR=${CODE_MOUNT}
MOUNTS="$CODE_DIR:$CODE_MOUNT,$OUTPUT_DIR:$OUTPUT_MOUNT,$DATA_DIR:$DATA_MOUNT"
# OUTFILE="${RESULTS_DIR}/slurm-%j-%n.out" # Can't be used with pty in srun
# ERRFILE="${RESULTS_DIR}/error-%j-%n.out"

GPU_LIMIT="$(($SLURM_GPUS_PER_NODE-1))"
SCRIPT_CUDA_VISIBLE_DEVICES=$(seq --separator=',' 0 $GPU_LIMIT)
SCRIPT_PYTHONPATH=${CODE_MOUNT}':$PYTHONPATH'

read -r -d '' RUN_COMMAND << EOF
echo '*******STARTING********' \
&& echo '---------------' \
&& wandb login ${WANDB} \
&& echo 'Starting training' \
&& export CUDA_VISIBLE_DEVICES=${SCRIPT_CUDA_VISIBLE_DEVICES} \
&& export PYTHONPATH=${SCRIPT_PYTHONPATH} \
&& export HYDRA_FULL_ERROR=1 \
&& cd ${CODE_MOUNT}/examples/chem \
&& python megamolbart_pretrain.py \
    --config-path=conf \
    --config-name=${MEGAMOLBART_CONFIG_FILE} \
    exp_manager.name=${EXP_DIR} \
    exp_manager.exp_dir=${RESULTS_MOUNT} \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    trainer.gpus=${SLURM_GPUS_PER_NODE} \
    tokenizer.vocab_path=${CODE_MOUNT}/nemo/collections/chem/vocab/megamolbart_pretrain_vocab.txt \
    model.train_ds.filepath=${DATA_MOUNT}/train/${DATA_FILES_SELECTED} \
    model.train_ds.metadata_path=${DATA_MOUNT}/train/metadata.txt \
    model.train_ds.batch_size=512 \
    model.train_ds.num_workers=10 \
    model.validation_ds.filepath=${DATA_MOUNT}/val/${DATA_FILES_SELECTED} \
    model.validation_ds.metadata_path=${DATA_MOUNT}/val/metadata.txt \
    model.validation_ds.batch_size=512 \
    model.validation_ds.num_workers=4
EOF


SCRIPT_PATH=${RESULTS_DIR}/job_script.sh
echo "${RUN_COMMAND}" > ${SCRIPT_PATH}
export SCRIPT_MOUNT=${RESULTS_MOUNT}/job_script.sh

# srun --output $OUTFILE --error $ERRFILE \
srun --pty \
--account ent_joc_model_mpnn_pyt \
--partition interactive \
--nodes ${SLURM_JOB_NUM_NODES} \
--ntasks ${NTASKS} \
--ntasks-per-node ${SLURM_GPUS_PER_NODE} \
--gpus-per-node ${SLURM_GPUS_PER_NODE} \
--container-image ${CONTAINER} \
--container-mounts ${MOUNTS} \
--container-workdir ${WORKDIR} \
--export WANDB=${WANDB} \
--export PYTHONPATH="${SCRIPT_PYTHONPATH}" \
--export RUN_COMMAND="${RUN_COMMAND}" \
--export SCRIPT_PATH="${SCRIPT_MOUNT}" \
--export TERM=xterm \
--nv-meta ml-model.megamolbart_int \
bash

# bash ${OUTPUT_MOUNT}/${EXP_DIR}/job_script.sh 
# bash -c "${RUN_COMMAND}"

set +x
