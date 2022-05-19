# MegaMolBART

MegaMolBART is a NeMo collection for large scale deep learning in cheminformatics with Megatron. [NeMo](https://github.com/NVIDIA/NeMo) is NVIDIA's toolkit for deep learning experimentation. 

## Quick Start

The [Quickstart Guide](QUICKSTART.md) contains

### Configure `launch.sh` script

The `launch.sh` script can be used to build the NeMo MegaMolBART training container, push it to a registry, and to automate the mounting of paths inside the container. Here is an example of the file. It should be named `.env` and placed inside the repo. All of the variables are described in the Usage section of `launch.sh` in this directory. Missing variables will be substituted for the defaults in the script.

```
MEGAMOLBART_CONT=nvcr.io/nvidian/clara-lifesciences/megamolbart_training_nemo:latest
PROJECT_PATH=$(pwd)
DATA_PATH=${HOME}/data
RESULT_PATH=${HOME}/results/nemo_experiments
GITHUB_ACCESS_TOKEN=INSERT_GITHUB_ACCESS_TOKEN_HERE
WANDB_API_KEY=$(grep password $HOME/.netrc | cut -d' ' -f4)
REGISTRY_ACCESS_TOKEN=$(grep apikey ~/.ngc/config | cut -d' ' -f3)
REGISTRY_USER=$oauthtoken
REGISTRY=NotSpecified
```

### Build container

The `launch.sh` script can be used to build and push containers to a registry. It can also be used to run interactive development jobs on a local system. See the instructions inside the script for more information. Once the `.env` script is created, a container can be built by running `bash launch.sh build`. If pushing to a registry is desired, `bash launch.sh dev` will complete this task.

### Run training job

#### Local machine (non-SLURM)

A desktop machine (or similar, that doesn't use SLURM) can be used for development and training. First, after building and pulling a copy of the Docker container to the computer, run `bash launch.sh dev` which will drop the user into a shell inside the container. After configuring the configuration files below (see also the presentation above), a job can be run using one of the scripts from inside the `examples/chem/nosched` directory. The `megamolbart_pretrain_quick.sh` script is a good way to quickly test a training run. This script should be executed INSIDE the docker container.

###  Edit or create model configuration file

MegaMolBART using Hydra/OmegaConf for configuration, so the parameters are yaml file. The existing model configuration files can be found in `examples/chem` and are based on the configurations for [Chemformer](https://chemrxiv.org/engage/chemrxiv/article-details/60ee8a3eb95bdd06d062074b):
* `megamolbart_pretrain_small_aug.yaml`: Small model with augmentation of the encoder SMILES
* `megamolbart_pretrain_small_span_aug.yaml`: Small model with augmentation of the encoder SMILES and masking of decoder tokens
* `megamolbart_pretrain_small_span.yaml`: Small model with masking of decoder tokens
* `megamolbart_pretrain_large_span_aug.yaml`: Extra large model with augmentation of the encoder SMILES and masking of decoder tokens
* `megamolbart_pretrain_xsmall_span_aug.yaml`: Extra small model with augmentation of the encoder SMILES and masking of decoder tokens, mainly for testing

Each of these master parameter files depends on a heirarchy of dependent yaml files found in the `examples/chem/conf` directory.Additional files can be created to suit other configurations. The master parameter files are read by the script `megamolbart_pretrain.py` which runs the training loop.

## Edit SLURM / computer execution script

The bash scripts found in `examples/chem/slurm` are configurable for the location of system directories for data, results, and (optionally) development code. These files are also used to configure the size of the training run (number of nodes and gpus). Note that MegaMolBART currently supports on data parallel training. 

For SLURM, once a working bash script has been created, consecutive training runs can be queued with the `auto_launcher.sh` script: `./auto_launcher.sh -n 5 megamolbart_pretrain_slurm.sh`.

## Conversion from CSV to NeMo format Binary Data
### In Beta phase! Use at your own risk
### Data Preprocessing
We support conversion of csv data to NeMo format binary data. The megamolbart_pretrain.py script can be used to preprocess the data into binary. 
Copy paste below fields into the *data* field of your config.
```
model:
  data:
    dataset_format: bin 
    num_enumerations: 5 #You can change this number of how many every enumerations you want on every SMILE string.
```
### Training
For training, make the following changes in the *data* field of your config
```
model:
  data:
    dataset_format: bin
    dataset_files: x[000..146]  
```
All files from 000 to 146 will be read for training. Do NOT add any extension to the data files here. The code looks for x[000...146].bin and x[000...146].idx on it's own. Giving an extension would mean that the code looks for x000.bin.bin and x000.bin.idx files, it will lead to File Not Found Errors. 

## Process and train with FULL default ZINC15 tranches dataset
Currently, the pretrain script uses a "test" default ZINC15 tranches data downloader text file to process and train. This is to prevent about 100GB of data being downloaded accidentally. If you wish to run with ALL of default ZINC15 tranches data, make the following changes.
Change the [line 113](https://github.com/clara-parabricks/NeMo_MegaMolBART/blob/dev/examples/chem/megamolbart_pretrain.py#L113) links_file path to this: 'conf/dataset/ZINC-downloader.txt. TODO: change the link when publishing.

Follow steps in Data Preprocessing and Training above if you want to preprocess the csv to binary and train with the binary files.
