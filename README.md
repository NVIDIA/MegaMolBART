# MegaMolBART

MegaMolBART is a NeMo collection for large scale deep learning in cheminformatics with Megatron. [NeMo](https://github.com/NVIDIA/NeMo) is NVIDIA's toolkit for deep learning experimentation. MegaMolBART itself

## Quick Start

See [this presentation](https://docs.google.com/presentation/d/1KdvYW5EGktie1k8xr8cym4gy_mIp4z1mumd1zWGXUow/edit#slide=id.p1) for a more in-depth overview of MegaMolBART in the NeMo framework.

### Configure `launch.sh` script

The `launch.sh` script can be used to build the NeMo MegaMolBART training container, push it to a registry, and to automate the mounting of paths inside the container. Here is an example of the file. It should be named `.cheminf_local_environment` and placed inside the repo. All of the variables are described in the Usage section of `launch.sh` in this directory. Missing variables will be substituted for the defaults in the script.

```
MEGAMOLBART_CONT=nvcr.io/nvidian/clara-lifesciences/megamolbart_training_nemo:210830
PROJECT_PATH=$(pwd)
JUPYTER_PORT=8988
DATA_PATH=/home/mgill/data
GITHUB_ACCESS_TOKEN=INSERT_GITHUB_ACCESS_TOKEN_HERE
WANDB_API_KEY=$(grep password $HOME/.netrc | cut -d' ' -f4)
REGISTRY_ACCESS_TOKEN=$(grep apikey ~/.ngc/config | cut -d' ' -f3)
REGISTRY_USER=$oauthtoken
REGISTRY=NotSpecified
RESULT_PATH=/home/mgill/results/nemo_experiments
```

### Build container

The `launch.sh` script can be used to build and push containers to a registry. It can also be used to run interactive development jobs on a local system. See the instructions inside the script for more information. Once the `.cheminf_local_environment` script is created, a container can be built by running `bash launch.sh build`. If pushing to a registry is desired, `bash launch.sh dev` will complete this task.

### Run training job

#### Local machine (non-SLURM)

A desktop machine (or similar, that doesn't use SLURM) can be used for development and training. First, after building and pulling a copy of the Docker container to the computer, run `bash launch.sh dev` which will drop the user into a shell inside the container. After configuring the configuration files below (see also the presentation above), a job can be run using one of the scripts from inside the `examples/chem/nosched` directory. The `megamolbart_pretrain_quick.sh` script is a good way to quickly test a training run. This script should be executed INSIDE the docker container.

###  Edit or create model configuration file

MegaMolBART using Hydra/OmegaConf for configuration, so the parameters are yaml file. The existing model configuration files can be found in `examples/chem` and are based on the configurations for [Chemformer](https://chemrxiv.org/engage/chemrxiv/article-details/60ee8a3eb95bdd06d062074b):
* `megamolbart_pretrain_small_aug.yaml`: Small model with augmentation of the encoder SMILES
* `megamolbart_pretrain_small_aug_span.yaml`: Small model with augmentation of the encoder SMILES and masking of decoder tokens
* `megamolbart_pretrain_small_span.yaml`: Small model with masking of decoder tokens
* `megamolbart_pretrain_xlarge_aug_span.yaml`: Extra large model with augmentation of the encoder SMILES and masking of decoder tokens
* `megamolbart_pretrain_xsmall_aug_span.yaml`: Extra small model with augmentation of the encoder SMILES and masking of decoder tokens, mainly for testing

Each of these master parameter files depends on a heirarchy of dependent yaml files found in the `examples/chem/conf` directory.Additional files can be created to suit other configurations. The master parameter files are read by the script `megamolbart_pretrain.py` which runs the training loop.

## Edit SLURM / computer execution script

The bash scripts found in `examples/chem/slurm` and `examples/chem/nosched` are configurable for the location of system directories for data, results, and (optionally) development code. These files are also used to configure the size of the training run (number of nodes and gpus). Note that MegaMolBART currently supports on data parallel training. 

For SLURM, once a working bash script has been created, consecutive training runs can be queued with the `auto_launcher.sh` script: `./auto_launcher.sh -n 4 /path/to/megamolbart_pretrain_slurm.sh`.

