#!/bin/bash
#SBATCH --nodes 4
#SBATCH --ntasks 32 
#SBATCH --ntasks-per-node 8 
#SBATCH --gpus-per-node 8 
#SBATCH --time=8:00:00
#SBATCH --partition batch
#SBATCH --account ent_joc_model_mpnn_pyt
#SBATCH --job-name megamolbart
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --overcommit            # Needed for pytorch
#SBATCH --gres=gpfs:circe       # Needed for Circe-Draco <required>

set -x

### CONFIG ###
DATA_FILES_SELECTED="x_OP_000..050_CL_.csv"

CONTAINER="nvcr.io#nvidian/clara-lifesciences/megamolbart_training_nemo:210716"
STORAGE_DIR="/gpfs/fs1/projects/ent_joc/users/mgill/megatron"
PROJECT="MegaMolBART" # exp_manager and wandb
EXPNAME="Draco-RNO" # exp_manager and wandb

WANDB=$(grep API_KEY ~/.config/wandb | cut -d' ' -f3)
DATA_DIR=${STORAGE_DIR}/data/zinc_csv_split
CODE_DIR=${STORAGE_DIR}/code/NeMo
OUTPUT_DIR=${STORAGE_DIR}/nemo

### 
RESULTS_DIR="${OUTPUT_DIR}/${EXPNAME}_nodes_${SLURM_JOB_NUM_NODES}_gpus_${SLURM_GPUS_PER_NODE}"
mkdir -p ${OUTPUT_DIR}
mkdir -p ${RESULTS_DIR}

DATA_MOUNT=/data
CODE_MOUNT=/code
OUTPUT_MOUNT=/output
WORKDIR=${CODE_MOUNT}
MOUNTS="$CODE_DIR:$CODE_MOUNT,$OUTPUT_DIR:$OUTPUT_MOUNT,$DATA_DIR:$DATA_MOUNT"
OUTFILE="${RESULTS_DIR}/slurm-%j-%n.out"
ERRFILE="${RESULTS_DIR}/error-%j-%n.out"

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
read -r -d '' RUN_COMMAND <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& wandb login ${WANDB} \
&& echo "Starting training" \
&& cd ${CODE_MOUNT} \
&& python examples/chem/draco-rno/megamolbart_pretrain_interactive.sh \
    --config-path=examples/chem/conf \
    --config-name=megatron_pretrain \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    trainer.gpus=${SLURM_GPUS_PER_NODE} \
    model.train_ds.filepath=/data/train/${DATA_FILES_SELECTED} \
    model.validation_ds.filepath=/data/val/${DATA_FILES_SELECTED} \
    exp_manager.wandb_logger_kwargs.name=${EXPNAME}_nodes_${SLURM_JOB_NUM_NODES}_gpus_${SLURM_GPUS_PER_NODE} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT} \
    exp_manager.exp_dir=${OUTPUT_MOUNT}
EOF

srun \
--mpi=pmix \
--container-image ${CONTAINER} \
--container-mounts ${MOUNTS} \
--container-workdir ${WORKDIR} \
--export PYTHONPATH="$PYTHONPATH"':$PYTHONPATH' \
--export RUN_COMMAND="$RUN_COMMAND" \
--export WANDB="$WANDB" \
--output $OUTFILE \
--error $ERRFILE \
bash -c "${RUN_COMMAND}"

set +x
