# Quickstart Guide

## Introduction

The following Quickstart guide contains configuration information and examples of how to run data processing and training of a small model on a workstation or SLURM-enabled cluster.

## Logging with Tensorboard and Weights and Biases

NeMo provides optional logging with Tensorboard and Weights and Biases. Use of Weights and Biases is optional, but highly recommended. All that is required is an account and an [API key](https://docs.wandb.ai/guides/track/public-api-guide).

## Docker Container

The latest docker container can be downloaded from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/megamolbart_v0.2/tags). It is also possible to build a container (see the [README](./README.md)), but using the provided container is **highly** recommended.

The following are best practices for mounting volumes for data, results, and code within the container:

- `/data` : the directory which will container the data (see Data Processing section) should be mounted within the container.
- `/result` : training results, including logs and checkpoints
- `/opt/nvidia/nemo_chem` : this is where the MegaMolBART code resides. It is already installed in the container, so a volume mount is not required. However, for development (see Code Development section below), it is convenient to have the code mounted so that modifications can be preserved between container restarts.

## Data Processing

MegaMolBART uses a subset of ZINC15 for training, as described in the [README](./README.md). The tranches can be downloaded and processed automatically.

A sample docker run command for processing data looks like the following, where `MEGAMOLBART_CONT` is edited to contain the name and tag of the container, and `DATA_PATH` is the location of the directory on the local machine which is mounted inside the container for the data.

```bash
#!/bin/bash

MEGAMOLBART_CONT=""
DATA_PATH=""

RUN_SCRIPT="cd ${HOME}/examples/chem && \
python megamolbart_pretrain.py \
--config-path=conf \
--config-name=megamolbart_pretrain_xsmall_span_aug \
++do_training=False \
++do_testing=False"

docker run -t \
--rm \
--network host \
--gpus all \
--shm-size=1g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--volume ${DATA_PATH}:/data \
--env HOME=/opt/nvidia/nemo_chem \
--workspace /opt/nvidia/nemo_chem \
--name megamolbart_data \
${MEGAMOLBART_CONT} \
bash -c $RUN_SCRIPT
```

In the script above, the config directory corresponds to the `examples/chem/conf` directory within the repo. The config name here is the name of the model yaml config file and is required, although model configuration parameters are not utilized for data processing.

## Training MegaMolBART

### Training on a Workstation

The easiest way to understand how training works is to train a very small version of the model. A sample docker run command for training on a workstation looks like the following, where `RESULT_PATH` is edited to where model training results are stored.

The `WANDB_API_KEY` variable contains a Weights & Biases API key or can be omitted if you do not have an account. The variable `WANDB_OFFLINE` controls whether or not the Weights and Biases log files are uploaded to a Weights and Biases account during training. **If `WANDB_OFFLINE` is set to `FALSE` and an invalid `WANDB_API_KEY` is provided, the training will hang before starting.** Thus, it is recommended to leave it set to `TRUE` when getting started with MegaMolBART. Any log files can also be uploaded after the training has finished, as described in the Weights and Biases documentation.

The remaining variables are as set previously.

```bash
#!/bin/bash

MEGAMOLBART_CONT=""
DATA_PATH=""
RESULT_PATH=""
WANDB_API_KEY=""
WANDB_OFFLINE="TRUE" # Set to FALSE to upload to WandB during training

RUN_SCRIPT="cd ${HOME}/examples/chem && \
python megamolbart_pretrain.py \
--config-path=conf \
--config-name=megamolbart_pretrain_xsmall_span_aug \
exp_manager.wandb_logger_kwargs.offline=${WANDB_OFFLINE}"

docker run -t \
--rm \
--network host \
--gpus all \
--shm-size=1g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--volume ${DATA_PATH}:/data \
--volume ${RESULT_PATH}:/result \
--env HOME=/opt/nvidia/nemo_chem \
--workspace /opt/nvidia/nemo_chem \
--env WANDB_API_KEY=$WANDB_API_KEY \
--name megamolbart_train
${MEGAMOLBART_CONT} \
bash -c $RUN_SCRIPT
```

NOTE: To make it more convenient, a basic shell script can be found [here](./examples/chem/shell/megamolbart_pretrain_quick.sh). The shell script is most suitable for beginners or for quick feature testing. It's not meant for extensive customization across multiple platforms.

### SLURM Jobs

A SLURM script for training on a cluster looks very similar to the above training script. The `SBATCH` commands must be configured as appropriate for the cluster on which the job will be executed.

```bash
#!/bin/bash
#SBATCH --nodes NUM_NODES
#SBATCH --ntasks-per-node NUM_GPUS
#SBATCH --mail-type=FAIL
#SBATCH --time=4:00:00
#SBATCH --partition PARTITION_NAME
#SBATCH --account ACCOUNT_NAME
#SBATCH --job-name JOB_NAME

set -x

MEGAMOLBART_CONT=""
DATA_PATH=""
RESULT_PATH=""
WANDB_API_KEY=""
WANDB_OFFLINE="TRUE" # Set to FALSE to upload to WandB during training

MOUNTS="$DATA_PATH:/data,$RESULT_PATH:/result"

srun \
--output /results/slurm-%j-%n.out \
--error /results/error-%j-%n.out \
--container-image ${MEGAMOLBART_CONT} \
--container-mounts ${MOUNTS} \
--container-workdir /opt/nvidia/nemo_chem \
--export WANDB_API_KEY="${WANDB_API_KEY}" \
python megamolbart_pretrain.py \
    --config-path=conf \
    --config-name=megamolbart_pretrain_xsmall_span_aug \
    ++trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    ++trainer.gpus=${SLURM_NTASKS_PER_NODE} \
    exp_manager.wandb_logger_kwargs.offline=${WANDB_OFFLINE}

set +x
```

## Code Development

A sample docker run command for interactive development might look like the following, where `DATA_PATH,` `RESULT_PATH`, and `PROJECT_PATH` are the respective paths on the local machine. The `WANDB_API_KEY` variable contains your API key or can be omitted if you do not have an account.

It is best practice to reinstall any development code before development or training by running the `pip install -e .` within the top level of the directory. It may also be necessary to recompile the Megatron helpers, which can be done right before running the container.

```bash
#!/bin/bash

MEGAMOLBART_CONT=""
DATA_PATH=""
RESULT_PATH=""
PROJECT_PATH=$(pwd)
WANDB_API_KEY=""
WANDB_OFFLINE="TRUE" # Set to FALSE to upload to WandB during training

docker run -it \
--rm \
--network host \
--gpus all \
--shm-size=1g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--volume ${DATA_PATH}:/data \
--volume ${RESULT_PATH}:/result \
--volume ${PROJECT_PATH}:/opt/nvidia/nemo_chem
--env HOME=/opt/nvidia/nemo_chem \
--workspace /opt/nvidia/nemo_chem \
--env WANDB_API_KEY=$WANDB_API_KEY \
--name megamolbart_dev
${MEGAMOLBART_CONT} \
cd ${HOME} && \
pip install -e . && \
bash setup/recompile_megatron_helper.sh && \
bash
```

## Training Output

### Log Files

NeMo creates a number of log files during training in the results directory:

- `cmd-args.log` : the python commands and any additional parameter overrides used for training
- `hparams.yam` : the final set of all hyperparmeters used for training
- `git-info.log`  : the commit hash of the code used for training and the results of `git diff` on this code to store any additional modifications from this commit
- `nemo_log_globalrank-0_localrank-0.txt`  : NeMo creates a number of logs, depending on how many GPUs are used in training. For < 32 GPUs, every GPU will will get is own log. When more than 32 GPUs are used, the number of logs is scaled down to one per node.
- `nemo_error_log.txt`  : error logs from NeMo
- `lightning_logs.txt`  : PyTorch Lightning logs

### Checkpoints

Checkpoints are stored in the `checkpoints` directory and are managed by NeMo. For example, it is possible to ensure only the top k checkpoints are saved. See the NeMo documentation for more information. NeMo can also optional create a `*.nemo` checkpoint which has the optimizer states removed and can be used for inference or fine tuning tasks. Checkpoints are automatically reloaded

### Tensorboard and Weights and Biases

NeMo will also create a Tensorboard file in the results directory, if logging to Tensorboard has been enabled. The Weights and Biases logs will be created in a directory called `wandb` and can optionally be uploaded after training if `WANDB_OFFLINE` was set to `TRUE` during training.