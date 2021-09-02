#!/bin/bash
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 8
#SBATCH --mail-type=FAIL
#SBATCH --time=4:00:00                          # Selene
#SBATCH --partition luna                        # Selene
#SBATCH --account swdl                          # Selene
#SBATCH --job-name swdl-clara:mgill-megamolbart # Selene

set -x

##### Development on a cluster with SLURM
### CONFIG ###

HOSTNAME=Selene

PROJECT=MegaMolBART
MEGAMOLBART_CONFIG_FILE=small_span_aug
DATA_FILES_SELECTED=x_OP_000..146_CL_.csv
CONTAINER="FIXME"

STORAGE_DIR=${HOME}/fs/megatron # ${HOME}/fs is a link to luster fs mount
WANDB_API_KEY=$(grep password $HOME/.netrc | cut -d' ' -f4)
DATA_DIR=${STORAGE_DIR}/data/zinc_csv_split
CODE_DIR=${STORAGE_DIR}/code/NeMo
OUTPUT_DIR=${STORAGE_DIR}/nemo

### END CONFIG ###

EXP_NAME=${HOSTNAME}_nodes_${SLURM_JOB_NUM_NODES}_gpus_${SLURM_NTASKS_PER_NODE}

RESULTS_DIR=${OUTPUT_DIR}/${PROJECT}/${MEGAMOLBART_CONFIG_FILE}/${EXP_NAME}
OUTFILE="${RESULTS_DIR}/slurm-%j-%n.out"
ERRFILE="${RESULTS_DIR}/error-%j-%n.out"

DATA_MOUNT=/data
CODE_MOUNT=/workspace/nemo
OUTPUT_MOUNT=/result
RESULTS_MOUNT=${OUTPUT_MOUNT}/${PROJECT}/${MEGAMOLBART_CONFIG_FILE}/${EXP_NAME}
WORKDIR=${CODE_MOUNT}

MOUNTS="$CODE_DIR:$CODE_MOUNT,$OUTPUT_DIR:$OUTPUT_MOUNT,$DATA_DIR:$DATA_MOUNT"

mkdir -p ${RESULTS_DIR}
GPU_LIMIT="$(($SLURM_NTASKS_PER_NODE-1))"
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
    exp_manager.wandb_logger_kwargs.offline=${WANDB_OFFLINE_MODE} \
    exp_manager.wandb_logger_kwargs.job_type=${EXP_NAME} \
    exp_manager.name=${EXP_NAME} \
    exp_manager.exp_dir=${RESULTS_MOUNT} \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    trainer.gpus=${SLURM_NTASKS_PER_NODE} \
    tokenizer.vocab_path=${CODE_MOUNT}/nemo/collections/chem/vocab/megamolbart_pretrain_vocab.txt \
    model.train_ds.filepath=${DATA_MOUNT}/train/${DATA_FILES_SELECTED} \
    model.train_ds.metadata_path=${DATA_MOUNT}/train/metadata.txt \
    model.validation_ds.filepath=${DATA_MOUNT}/val/${DATA_FILES_SELECTED} \
    model.validation_ds.metadata_path=${DATA_MOUNT}/val/metadata.txt \
    model.train_ds.batch_size=128 \
    model.validation_ds.batch_size=128
EOF

SCRIPT_PATH=${RESULTS_DIR}/job_script.sh
echo "${RUN_COMMAND}" > ${SCRIPT_PATH}
export SCRIPT_MOUNT=${RESULTS_MOUNT}/job_script.sh

srun $ADDITIONAL_FLAGS \
--output $OUTFILE --error $ERRFILE \
--container-image ${CONTAINER} \
--container-mounts ${MOUNTS} \
--container-workdir ${WORKDIR} \
--export PYTHONPATH="${SCRIPT_PYTHONPATH}" \
--export SCRIPT_PATH="${SCRIPT_MOUNT}" \
--export WANDB_API_KEY="${WANDB_API_KEY}" \
bash ${SCRIPT_MOUNT}

set +x