#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8      # n gpus per machine <required>
#SBATCH --mail-type=FAIL
#SBATCH --time=8:00:00
#SBATCH --partition=batch_dgx1_m2
#SBATCH --account=ent_aiapps_omics
#SBATCH --job-name=bionemo
#SBATCH --nv-meta=ml-model.megamolbart
#SBATCH --mem=0                 # all mem avail
#SBATCH --overcommit
#SBATCH --exclusive             # exclusive node access

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

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--prop)
            echo 'Overwriting values from $2.'
            PROPERTY_FILES=$2
            shift
            shift
            ;;
        *)
            echo 'Invalid input'
            exit 1
            ;;
    esac
done

# All variables with default values must be defined in this section
#=========================
MEGAMOLBART_CONT="nvcr.io#t6a4nuz8vrsr/megamolbart:0.2.0-ea3"
STORAGE_ROOT=""
WANDB_API_KEY=""
MICRO_BATCH_SIZE=256 # Please check GPU mem size. 256 is recommended for A100 with 80 GB mem.
VAL_CHECK_INTERVAL=200
JOB_TYPE='nemo-chem'
EXP_NAME_PREFIX='nemo_chem'
#=========================

set -e
# Any value that needs to be overwritten should be defined in PROPERTY_FILES
if [ ! -z "${PROPERTY_FILES}" ];
then
    IFS=',' read -ra FILES <<< ${PROPERTY_FILES}
    for PROPERTY_FILE in "${FILES[@]}"; do
        source ${PROPERTY_FILE}
    done
fi

if [ -z "${STORAGE_ROOT}" ];
then
    echo "STORAGE_ROOT is invaild. STORAGE_ROOT=${STORAGE_ROOT}. Please check the properties file."
    exit 1
fi

EXP_NAME=${EXP_NAME_PREFIX}_node_${SLURM_JOB_NUM_NODES}_inv${VAL_CHECK_INTERVAL}
DATA_PATH="${STORAGE_ROOT}/data"
RESULT_PATH="${STORAGE_ROOT}/results/${EXP_NAME}"
MOUNTS="$DATA_PATH:/data,$RESULT_PATH:/result"
mkdir -p ${RESULT_PATH}

# This configuration assumes TP and PP is 1.
# TODO: Revisit to apply the following calculation
#       global_batch = micro_batch * TP * PP * DP
GLOBAL_BATCH_SIZE=$(expr ${MICRO_BATCH_SIZE} \* ${SLURM_JOB_NUM_NODES} \* ${SLURM_NTASKS_PER_NODE})


# NeMo and BioNeMo code is picked from the container. To use code from a shared
# folder instead, please NEMO_CODE and BIONEMO_CODE in the properties file.
if [ ! -z "${NEMO_CODE}" ];
then
    MOUNTS="${MOUNTS},${NEMO_CODE}:/opt/nvidia/nemo"
fi

if [ ! -z "${BIONEMO_CODE}" ];
then
    MOUNTS="${MOUNTS},${BIONEMO_CODE}:/opt/nvidia/nemo_chem"
fi

set -x
srun \
    --output slurm-%j-%n.out \
    --error error-%j-%n.out \
    --container-image ${MEGAMOLBART_CONT} \
    --container-mounts ${MOUNTS} \
    --container-workdir /opt/nvidia/nemo_chem/examples/chem/ \
    --export WANDB_API_KEY="${WANDB_API_KEY}" \
    python megamolbart_pretrain.py \
        --config-path=conf \
        --config-name=megamolbart_pretrain_small_span_aug \
        ++exp_manager.wandb_logger_kwargs.job_type="${JOB_TYPE}" \
        ++exp_manager.wandb_logger_kwargs.name="${EXP_NAME}"\
        ++trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
        ++trainer.gpus=${SLURM_NTASKS_PER_NODE} \
        ++trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
        ++trainer.max_steps=20000000 \
        model.micro_batch_size=${MICRO_BATCH_SIZE} \
        model.global_batch_size=${GLOBAL_BATCH_SIZE} \
        model.tokenizer.model=/opt/nvidia/nemo_chem/models/vocab/megamolbart.model \
        model.tokenizer.vocab_file=/opt/nvidia/nemo_chem/models/vocab/megamolbart.vocab \
        model.data.links_file=/opt/nvidia/nemo_chem/examples/chem/conf/dataset/ZINC-downloader-test.txt \
        model.data.dataset.val=x000-small

set +x
