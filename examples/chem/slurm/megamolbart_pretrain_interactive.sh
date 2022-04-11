#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -x

##### Development on a cluster with SLURM / Optional interactive or batch training
### CONFIG ###

HOSTNAME=Hostname
ENCODER_TYPE=${1:-seq2seq}

SLURM_JOB_NUM_NODES=1 # These are used for interactive jobs for consistency with SLURM scripts
SLURM_TASKS_PER_NODE=8

ADDITIONAL_FLAGS=" --time 2:00:00 --partition PARTITION_NAME --account ACCOUNT_NAME --job-name JOB_NAME "
IS_BATCH=0 # 0 for interactive, 1 for sbatch

PROJECT=MegaMolBART
MEGAMOLBART_CONFIG_FILE=small_span_aug
DATA_FILES_SELECTED=x_OP_000..001_CL_.csv
CONTAINER="nvcr.io#nvidian/clara-lifesciences/megamolbart_training_nemo:latest" # TODO update with public path

STORAGE_DIR=${HOME}/megatron
WANDB_API_KEY=$(grep password $HOME/.netrc | cut -d' ' -f4)
DATA_DIR=${STORAGE_DIR}/data/zinc_csv_split
CODE_DIR=${STORAGE_DIR}/code/nemo_chem
OUTPUT_DIR=${STORAGE_DIR}/nemo

### END CONFIG ###

EXP_NAME=${HOSTNAME}_nodes_${SLURM_JOB_NUM_NODES}_gpus_${SLURM_TASKS_PER_NODE}
RESULTS_DIR=${OUTPUT_DIR}/${PROJECT}/${MEGAMOLBART_CONFIG_FILE}/${EXP_NAME}
OUTFILE="${RESULTS_DIR}/slurm-%j-%n.out" # Ignored in interactive mode
ERRFILE="${RESULTS_DIR}/error-%j-%n.out" # Ignored in interactive mode

DATA_MOUNT=/data
CODE_MOUNT=/workspace/nemo_chem
OUTPUT_MOUNT=/result
RESULTS_MOUNT=${OUTPUT_MOUNT}/${PROJECT}/${MEGAMOLBART_CONFIG_FILE}/${EXP_NAME}
WORKDIR=${CODE_MOUNT}

MOUNTS="$CODE_DIR:$CODE_MOUNT,$OUTPUT_DIR:$OUTPUT_MOUNT,$DATA_DIR:$DATA_MOUNT"

mkdir -p ${RESULTS_DIR}
GPU_LIMIT="$(($SLURM_TASKS_PER_NODE-1))"
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
&& export WANDB_API_KEY=${WANDB_API_KEY} \
&& python megamolbart_pretrain.py \
    --config-path=conf \
    --config-name=megamolbart_pretrain_${MEGAMOLBART_CONFIG_FILE} \
    dataset_path=${DATA_MOUNT} \
    exp_manager.wandb_logger_kwargs.offline=${WANDB_OFFLINE_MODE} \
    exp_manager.wandb_logger_kwargs.job_type=${EXP_NAME} \
    exp_manager.name=${EXP_NAME} \
    exp_manager.exp_dir=${RESULTS_MOUNT} \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    trainer.gpus=${SLURM_TASKS_PER_NODE} \
    model.train_ds.filepath=${DATA_MOUNT}/train/${DATA_FILES_SELECTED} \
    model.validation_ds.filepath=${DATA_MOUNT}/val/${DATA_FILES_SELECTED} \
    model.train_ds.micro_batch_size=128 \
    model.validation_ds.micro_batch_size=128 \
    model.encoder_type=${ENCODER_TYPE} \
    ++tokenizer.vocab_path=${CODE_MOUNT}/nemo_chem/vocab/megamolbart_vocab.txt \
    ++trainer.val_check_interval=0.5 \
    ++trainer.limit_val_batches=2 \
    ++trainer.limit_train_batches=10 \
    ++trainer.max_epochs=10 
EOF

SCRIPT_PATH=${RESULTS_DIR}/job_script.sh
echo "${RUN_COMMAND}" > ${SCRIPT_PATH}
export SCRIPT_MOUNT=${RESULTS_MOUNT}/job_script.sh

if [ ${IS_BATCH} -eq 0 ]; then
    ADDITIONAL_FLAGS=${ADDITIONAL_FLAGS}" --pty --nodes ${SLURM_JOB_NUM_NODES} --ntasks-per-node ${SLURM_TASKS_PER_NODE} "
    EXEC_COMMAND=" bash"
else
    ADDITIONAL_FLAGS="--output $OUTFILE --error $ERRFILE "
    # EXEC_COMMAND=" bash -c ${RUN_COMMAND}"
    EXEC_COMMAND=" bash ${SCRIPT_MOUNT}"
fi

srun $ADDITIONAL_FLAGS \
--container-image ${CONTAINER} \
--container-mounts ${MOUNTS} \
--container-workdir ${WORKDIR} \
--export PYTHONPATH="${SCRIPT_PYTHONPATH}" \
--export SCRIPT_PATH="${SCRIPT_MOUNT}" \
--export WANDB_API_KEY="${WANDB_API_KEY}" \
${EXEC_COMMAND}

set +x
