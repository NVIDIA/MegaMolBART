#!/bin/bash
set -x

# Tested with single node, multiple GPU configuration

### CONFIG ###
NUM_GPUS=8
NUM_NODES=1

DATA_FILES_SELECTED=x_OP_000..001_CL_.csv
PROJECT=MegaMolBART
EXPNAME=BENCHMARK_NGC
EXP_DIR=${EXPNAME}_nodes_${NUM_NODES}_gpus_${NUM_GPUS}

WANDB=88800d16aea5891a1cdab809b2c47c351c8125e1
DATA_MOUNT=/data/zinc_csv
CODE_MOUNT=/code/NeMo
OUTPUT_MOUNT=/result/nemo_experiments

mkdir -p ${OUTPUT_MOUNT}
GPU_LIMIT="$(($NUM_GPUS-1))"
SCRIPT_CUDA_VISIBLE_DEVICES=$(seq --separator=',' 0 $GPU_LIMIT)

export CUDA_VISIBLE_DEVICES=${SCRIPT_CUDA_VISIBLE_DEVICES}
export PYTHONPATH=${CODE_MOUNT}:$PYTHONPATH
export HYDRA_FULL_ERROR=1
cd ${CODE_MOUNT}/examples/chem

wandb login ${WANDB}
python megamolbart_pretrain.py \
    --config-path=conf \
    --config-name=megamolbart_pretrain \
    exp_manager.name=${EXP_DIR} \
    exp_manager.exp_dir=${RESULTS_MOUNT} \
    trainer.num_nodes=1 \
    trainer.gpus=${NUM_GPUS} \
    tokenizer.vocab_path=${CODE_MOUNT}/nemo/collections/chem/vocab/megamolbart_pretrain_vocab.txt \
    model.validation_ds.filepath=${DATA_MOUNT}/val/${DATA_FILES_SELECTED} \
    model.validation_ds.metadata_path=${DATA_MOUNT}/val/metadata.txt \
    model.validation_ds.batch_size=512 \
    model.validation_ds.num_workers=10 \
    model.validation_ds.use_iterable=false \
    model.train_ds.filepath=${DATA_MOUNT}/train/${DATA_FILES_SELECTED} \
    model.train_ds.metadata_path=${DATA_MOUNT}/train/metadata.txt \
    model.train_ds.batch_size=512 \
    model.train_ds.num_workers=10 \
    model.train_ds.use_iterable=false \
    exp_manager.create_tensorboard_logger=false \
    exp_manager.create_wandb_logger=false \
    exp_manager.create_checkpoint_callback=false \
    trainer.max_steps=10000 \
    ~trainer.val_check_interval \
    +trainer.limit_val_batches=0.0 \
    +trainer.log_every_n_steps=200

set +x
