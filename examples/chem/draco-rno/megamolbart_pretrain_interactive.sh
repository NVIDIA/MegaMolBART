#!/bin/bash
set -x

##### Interactive training / development on a cluster with SLURM
# Tested with single node, multiple GPU configuration

### CONFIG ###
SLURM_JOB_NUM_NODES=1
SLURM_GPUS_PER_NODE=2
DATA_FILES_SELECTED="x_OP_000..001_CL_.csv"

CONTAINER="nvcr.io#nvidian/clara-lifesciences/megamolbart_training_nemo:210716"
STORAGE_DIR="/gpfs/fs1/projects/ent_joc/users/mgill/megatron"
PROJECT="MegaMolBART" # exp_manager and wandb
EXPNAME="Draco-RNO" # exp_manager and wandb

WANDB=88800d16aea5891a1cdab809b2c47c351c8125e1
DATA_DIR=${STORAGE_DIR}/data/zinc_csv_split
CODE_DIR=${STORAGE_DIR}/code/NeMo
OUTPUT_DIR=${STORAGE_DIR}/nemo

### 
RESULTS_DIR="${OUTPUT_DIR}/${EXPNAME}_nodes_${SLURM_JOB_NUM_NODES}_gpus_${SLURM_GPUS_PER_NODE}"
mkdir -p ${RESULTS_DIR}

DATA_MOUNT=/data
CODE_MOUNT=/code
OUTPUT_MOUNT=/result
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
    --config-name=megamolbart_pretrain \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    trainer.gpus=${SLURM_GPUS_PER_NODE} \
    tokenizer.vocab_path=${CODE_MOUNT}/nemo/collections/chem/vocab/megamolbart_pretrain_vocab.txt \
    model.train_ds.filepath=/data/train/${DATA_FILES_SELECTED} \
    model.validation_ds.filepath=/data/val/${DATA_FILES_SELECTED} \
    exp_manager.wandb_logger_kwargs.name=${EXPNAME}_nodes_${SLURM_JOB_NUM_NODES}_gpus_${SLURM_GPUS_PER_NODE} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT} \
    exp_manager.exp_dir=${OUTPUT_MOUNT}/${EXPNAME}_nodes_${SLURM_JOB_NUM_NODES}_gpus_${SLURM_GPUS_PER_NODE}
EOF

echo "${RUN_COMMAND}" > ${RESULTS_DIR}/job_script.sh

# srun --output $OUTFILE --error $ERRFILE \
srun --pty \
--account ent_joc_model_mpnn_pyt \
--partition interactive \
--mpi pmix \
--nodes ${SLURM_JOB_NUM_NODES} \
--ntasks ${SLURM_GPUS_PER_NODE} \
--ntasks-per-node ${SLURM_GPUS_PER_NODE} \
--gpus-per-node ${SLURM_GPUS_PER_NODE} \
--container-image ${CONTAINER} \
--container-mounts ${MOUNTS} \
--container-workdir ${WORKDIR} \
--export WANDB=${WANDB} \
--export PYTHONPATH="${SCRIPT_PYTHONPATH}" \
--export RUN_COMMAND="${RUN_COMMAND}" \
bash
# bash ${OUTPUT_MOUNT}/${EXPNAME}_nodes_${SLURM_JOB_NUM_NODES}_gpus_${SLURM_GPUS_PER_NODE}/job_script.sh 
# bash -c "${RUN_COMMAND}"

set +x
