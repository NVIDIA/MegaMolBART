#!/bin/bash
set -x

##### Interactive training / development on a cluster with SLURM
# Tested only with single node, single GPU configuration

### CONFIG ###
SLURM_JOB_NUM_NODES=1
SLURM_GPUS_PER_NODE=2
WANDB=$(grep API_KEY ~/.config/wandb | cut -d' ' -f3)

CONTAINER="nvidian/clara-lifesciences/megamolbart_training_nemo:210716"
STORAGE_DIR="/gpfs/fs1/projects/ent_joc/users/mgill/megatron"
PROJECT="MegaMolBART" # exp_manager and wandb
EXPNAME="Draco-RNO" # exp_manager and wandb

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
WORKDIR=${CODE_DIR}
MOUNTS="$CODE_DIR:$CODE_MOUNT,$OUTPUT_DIR:$OUTPUT_MOUNT,$DATA_DIR:$DATA_MOUNT"
OUTFILE="${OUTPUT_DIR}/slurm-%j-%n.out"
ERRFILE="${OUTPUT_DIR}/error-%j-%n.out"

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
	model.train_ds.filepath='/data/train/x_OP_000..001_CL_.csv \
	model.validation_ds.filepath='/data/val/x_OP_000..001_CL_.csv \
	exp_manager.wandb_logger_kwargs.name=${EXPNAME}_nodes_${SLURM_JOB_NUM_NODES}_gpus_${SLURM_GPUS_PER_NODE} \
	exp_manager.wandb_logger_kwargs.project=${PROJECT} \
	exp_manager.exp_dir=${OUTPUT_MOUNT}
EOF

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash #-c "${RUN_COMMAND}"

srun \
--pty \
--account ent_joc_model_mpnn_pyt \
--partition interactive \
--export=PYTHONPATH="$PYTHONPATH"':$PYTHONPATH' \
--export=RUN_COMMAND="$RUN_COMMAND" \
--mpi=pmix \
--nodes ${SLURM_JOB_NUM_NODES} \
--ntasks ${SLURM_GPUS_PER_NODE} \
--ntasks-per-node ${SLURM_GPUS_PER_NODE} \
--gpus-per-node ${SLURM_GPUS_PER_NODE} \
--container-image ${CONTAINER} \
--container-mounts ${MOUNTS} \
--container-workdir ${WORKDIR} \
bash

set +x
