# MegaMolBART

MegaMolBART is a NeMo collection for large scale deep learning in cheminformatics with Megatron. [NeMo](https://github.com/NVIDIA/NeMo) is NVIDIA's toolkit for deep learning experimentation. MegaMolBART itself

## Quick Start

See [this presentation](https://docs.google.com/presentation/d/1KdvYW5EGktie1k8xr8cym4gy_mIp4z1mumd1zWGXUow/edit#slide=id.p1) for a more in-depth overview of MegaMolBART in the NeMo framework.

### Build container

The `launch.sh` script can be used to build and push containers to a registry. It can also be used to run interactive development jobs on a local system. See the instructions inside the script for more information.
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

