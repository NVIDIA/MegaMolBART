#!/bin/bash
set -x

# Tested with single node, multiple GPU configuration

### CONFIG ###
NUM_GPUS=3
NUM_NODES=1

DATA_FILES_SELECTED="x_OP_000..001_CL_.csv"
PROJECT=MegaMolBART
EXPNAME="BENCHMARK_NGC_nodes_${NUM_NODES}_gpus_${NUM_GPUS}"

WANDB=88800d16aea5891a1cdab809b2c47c351c8125e1
DATA_MOUNT=/data/zinc_csv_split
CODE_MOUNT=/code/NeMo
OUTPUT_MOUNT=/result/nemo_experiments

GPU_LIMIT=$(($NUM_GPUS-1))
SCRIPT_CUDA_VISIBLE_DEVICES=$(seq --separator=',' 0 $GPU_LIMIT)

export CUDA_VISIBLE_DEVICES=${SCRIPT_CUDA_VISIBLE_DEVICES}
export PYTHONPATH=${CODE_MOUNT}:$PYTHONPATH
wandb login ${WANDB}
export HYDRA_FULL_ERROR=1
cd ${CODE_MOUNT}/examples/chem

python megamolbart_pretrain.py \
    --config-path=conf \
    --config-name=megamolbart_pretrain \
    trainer.num_nodes=${NUM_NODES} \
    trainer.gpus=${NUM_GPUS} \
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

set +x
