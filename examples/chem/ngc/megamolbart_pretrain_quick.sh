#!/bin/bash
set -x

# Tested with single node, multiple GPU configuration

### CONFIG ###
NUM_GPUS=3
NUM_NODES=1
MEGAMOLBART_CONFIG_FILE=megamolbart_pretrain_xsmall_span_aug
DATA_FILES_SELECTED=x_OP_000..001_CL_.csv
CONTAINER="nvcr.io#nvidian/clara-lifesciences/megamolbart_training_nemo:210716"
WANDB=88800d16aea5891a1cdab809b2c47c351c8125e1
STORAGE_DIR=/gpfs/fs1/projects/ent_joc/users/mgill/megatron

PROJECT=MegaMolBART # exp_manager and wandb
EXPNAME=Draco-RNO # exp_manager and wandb
EXP_DIR=${EXPNAME}_nodes_${SLURM_JOB_NUM_NODES}_gpus_${SLURM_GPUS_PER_NODE}

DATA_MOUNT=/data/zinc_csv_split
CODE_MOUNT=/code/NeMo
OUTPUT_MOUNT=/result/nemo_experiments

mkdir -p ${OUTPUT_MOUNT}
GPU_LIMIT=$(($NUM_GPUS-1))
SCRIPT_CUDA_VISIBLE_DEVICES=$(seq --separator=',' 0 $GPU_LIMIT)

export CUDA_VISIBLE_DEVICES=${SCRIPT_CUDA_VISIBLE_DEVICES}
export PYTHONPATH="${CODE_MOUNT}:"$PYTHONPATH
export HYDRA_FULL_ERROR="1"
cd ${CODE_MOUNT}/examples/chem

# wandb login ${WANDB}
python megamolbart_pretrain.py \
    --config-path=conf \
    --config-name=${MEGAMOLBART_CONFIG_FILE} \
    exp_manager.name=${EXP_DIR} \
    exp_manager.exp_dir=${RESULTS_MOUNT} \
    trainer.num_nodes=${NUM_NODES} \
    trainer.gpus=${NUM_GPUS} \
    tokenizer.vocab_path=${CODE_MOUNT}/nemo/collections/chem/vocab/megamolbart_pretrain_vocab.txt \
    model.validation_ds.filepath=${DATA_MOUNT}/val/${DATA_FILES_SELECTED} \
    model.validation_ds.metadata_path=${DATA_MOUNT}/val/metadata.txt \
    model.validation_ds.batch_size=128 \
    model.validation_ds.num_workers=4 \
    model.train_ds.filepath=${DATA_MOUNT}/test/${DATA_FILES_SELECTED} \
    model.train_ds.metadata_path=${DATA_MOUNT}/test/metadata.txt \
    model.train_ds.batch_size=128 \
    model.train_ds.num_workers=4 \
    trainer.val_check_interval=1.0 \
    +trainer.limit_val_batches=2 \
    +trainer.limit_train_batches=4 \
    +trainer.max_epochs=2 

set +x
