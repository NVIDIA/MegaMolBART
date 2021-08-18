#!/bin/bash
#SBATCH --nodes 1 
#SBATCH --ntasks 16
#SBATCH --ntasks-per-node 16
#SBATCH --gpus-per-node 16
#SBATCH --time=8:00:00
#SBATCH --partition batch
#SBATCH --account ent_joc_model_mpnn_pyt
#SBATCH --nv-meta ml-model.megamolbart_benchmark
#SBATCH --mem=0                 # all mem avail
#SBATCH --exclusive

set -x

### CONFIG ###
DATA_FILES_SELECTED=x_OP_000..015_CL_.csv

CONTAINER="nvcr.io#nvidian/clara-lifesciences/megamolbart_training_nemo:210716"
STORAGE_DIR=/gpfs/fs1/projects/ent_joc/users/mgill/megatron
PROJECT=MegaMolBART # exp_manager and wandb
EXPNAME=Draco-RNO # exp_manager and wandb

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
RESULTS_MOUNT=${OUTPUT_MOUNT}/${EXPNAME}_nodes_${SLURM_JOB_NUM_NODES}_gpus_${SLURM_GPUS_PER_NODE}
WORKDIR=${CODE_MOUNT}
MOUNTS="$CODE_DIR:$CODE_MOUNT,$OUTPUT_DIR:$OUTPUT_MOUNT,$DATA_DIR:$DATA_MOUNT"
OUTFILE="${RESULTS_DIR}/slurm-%j-%n.out" # Can't be used with pty in srun
ERRFILE="${RESULTS_DIR}/error-%j-%n.out"

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
    model.validation_ds.filepath=${DATA_MOUNT}/val/${DATA_FILES_SELECTED} \
    model.validation_ds.metadata_path=${DATA_MOUNT}/val/metadata.txt \
    model.validation_ds.batch_size=1 \
    ~model.validation_ds.num_workers \
    model.validation_ds.use_iterable=false \
    model.train_ds.filepath=${DATA_MOUNT}/train/${DATA_FILES_SELECTED} \
    model.train_ds.metadata_path=${DATA_MOUNT}/train/metadata.txt \
    model.train_ds.batch_size=512 \
    model.train_ds.num_workers=20 \
    model.train_ds.use_iterable=false \
    exp_manager.create_tensorboard_logger=false \
    exp_manager.create_wandb_logger=false \
    exp_manager.create_checkpoint_callback=false \
    trainer.max_steps=10000 \
    ~trainer.val_check_interval \
    +trainer.limit_val_batches=0.0 \
    +trainer.log_every_n_steps=200
EOF

# model.train_ds.filepath=${DATA_MOUNT}/test/${DATA_FILES_SELECTED} \
# model.train_ds.metadata_path=${DATA_MOUNT}/test/metadata.txt \
# ~trainer.max_steps \
# +trainer.max_epochs=2 \
# ~trainer.val_check_interval \
# +trainer.limit_val_batches=2

echo "${RUN_COMMAND}" > ${RESULTS_DIR}/job_script.sh

srun \
--output $OUTFILE \
--error $ERRFILE \
--container-image ${CONTAINER} \
--container-mounts ${MOUNTS} \
--container-workdir ${WORKDIR} \
--export WANDB=${WANDB} \
--export PYTHONPATH="${SCRIPT_PYTHONPATH}" \
--export RUN_COMMAND="${RUN_COMMAND}" \
bash ${OUTPUT_MOUNT}/${EXPNAME}_nodes_${SLURM_JOB_NUM_NODES}_gpus_${SLURM_GPUS_PER_NODE}/job_script.sh 
# bash -c "${RUN_COMMAND}"

set +x
