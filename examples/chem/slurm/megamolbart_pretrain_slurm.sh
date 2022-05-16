#!/bin/bash
#SBATCH --nodes NUM_NODES
#SBATCH --ntasks-per-node NUM_GPUS
#SBATCH --mail-type=FAIL
#SBATCH --time=4:00:00
#SBATCH --partition PARTITION_NAME
#SBATCH --account ACCOUNT_NAME
#SBATCH --job-name JOB_NAME

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

MEGAMOLBART_CONT=""
DATA_PATH=""
RESULT_PATH=""
WANDB_API_KEY=""

MOUNTS="$DATA_PATH:/data,$RESULT_PATH:/result"

srun \
--output /results/slurm-%j-%n.out \
--error /results/error-%j-%n.out \
--container-image ${MEGAMOLBART_CONT} \
--container-mounts ${MOUNTS} \
--container-workdir /workspace/nemo_chem \
--export WANDB_API_KEY="${WANDB_API_KEY}" \
python megamolbart_pretrain.py \
    --config-path=conf \
    --config-name=megamolbart_pretrain_xsmall_span_aug \
    ++trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    ++trainer.gpus=${SLURM_NTASKS_PER_NODE}

set +x
