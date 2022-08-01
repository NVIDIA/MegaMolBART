#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8      # n gpus per machine <required>
#SBATCH --mail-type=FAIL
#SBATCH --time=8:00:00
#SBATCH --partition=batch_dgx1_m2
#SBATCH --account=ent_aiapps_omics
#SBATCH --job-name=bionemo-ea2-untied-token_head
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

MEGAMOLBART_CONT="gitlab-master.nvidia.com#mlivne/nemo_containers:megamolbart-r1.10.0-ea2-untied-weights"
DATA_PATH="/gpfs/fs1/projects/ent_aiapps/users/rilango/bionemo/data"
RESULT_PATH="/gpfs/fs1/projects/ent_aiapps/users/mlivne/results/bionemo/2022-07-12-ea2-untied"
WANDB_API_KEY=""

MOUNTS="$DATA_PATH:/data,$RESULT_PATH:/result"

mkdir -p ${RESULT_PATH}

srun \
--output ${RESULT_PATH}/slurm-%j-%n.out \
--error ${RESULT_PATH}/error-%j-%n.out \
--container-image ${MEGAMOLBART_CONT} \
--container-mounts ${MOUNTS} \
--container-workdir /opt/nvidia/nemo_chem/examples/chem/ \
--export WANDB_API_KEY="${WANDB_API_KEY}" \
python megamolbart_pretrain.py \
    --config-path=conf \
    --config-name=megamolbart_pretrain_small_span_aug \
    ++trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    ++trainer.gpus=${SLURM_NTASKS_PER_NODE} \
    model.global_batch_size=32 \
    model.micro_batch_size=4 \
    model.tokenizer.model=/opt/nvidia/nemo_chem/models/vocab/megamolbart.model \
    model.tokenizer.vocab_file=/opt/nvidia/nemo_chem/models/vocab/megamolbart.vocab \
    model.data.links_file=/opt/nvidia/nemo_chem/examples/chem/conf/dataset/ZINC-downloader-test.txt

set +x
