# README

## Introduction

MegaMolBART is a deep learning model for small molecule drug discovery and cheminformatics based on SMILES. MegaMolBART uses NVIDIA's [NeMo-Megatron](https://github.com/NVIDIA/NeMo) framework, which is designed for the development of large transformer models. More information about MegaMolBART is available in the [model guide](../../docs/ngc/model.md).

MegaMolBART relies on [NeMo](https://github.com/NVIDIA/NeMo). NeMo provides a robust environment for developing, training, and deploying deep learning models, including Megatron models. NeMo provides enhancements to PyTorch Lighting such as hyperparameter configurability with yaml files and checkpoint management. It also enables the development and training of large transformer models using NVIDIA's Megatron framework, which makes multi-GPU, multi-node training with data parallelism, model parallelism, and mixed precision easily configurable. The [NeMo User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/) contains more information about all of these features.

The ZINC-15 database is used for pre-training [1]. Approximately 1.45 Billion molecules (SMILES strings) were selected from [tranches](https://zinc15.docking.org/tranches/home/) meeting the following constraints: molecular weight <= 500 Daltons, LogP <= 5, reactivity level was "reactive", and purchasability was "annotated". SMILES formats, including chirality notations, are used as-is from ZINC.

During pre-processing, the compounds are filtered to ensure a maximum length of 512 characters. Train, validation, and test splits are randomly split using a seed as 99% / 0.5% / 0.5%. Data canonicalization and augmentation during training are performed using RDKit via masking and SMILES randomization as described previously [2].

The [Quickstart Guide](./QUICKSTART.md) contains configuration information and examples of how to run data processing and training of a small model on a workstation or SLURM-enabled cluster. The tutorial contained in the Quickstart is highly recommended to gain familiarity with how trainings are run and configured. The remainder of this README contains additional information that will be of use for more advanced tasks, such as code development or model configuration changes.

## Development Guide
### Configure `launch.sh` Script

The [`launch.sh` script](./launch.sh) can be used to build the NeMo MegaMolBART training container, push it to a registry, and to automate the mounting of paths inside the container. The script requires a settings file called `.env`. This file will automatically be created if it does not exist on first launch, but below is an example of the file. If created manually, it should be named `.env` and placed in the root of source path. All of the variables are described in the `usage` section of [`launch.sh`](./launch.sh) in this directory. Missing variables will be substituted for the defaults in the script.

```
# Docker Image used for training, inference, and development
MEGAMOLBART_CONT=nvcr.io/nvidia/clara/megamolbart:0.2.0

# Path to Source code
PROJECT_PATH=/home/rilango/Projects/github/NeMo_MegaMolBART/

# Path to data and results file
DATA_PATH=/data/bionemo/data/
RESULT_PATH=/data/bionemo/results/

# HOST port for jupyter lab web interface
JUPYTER_PORT=8888

# Docker image repository configuration
REGISTRY_USER=$oauthtoken
REGISTRY=NotSpecified
REGISTRY_ACCESS_TOKEN=$(grep apikey ~/.ngc/config | cut -d' ' -f3)

# wandb API key. ttps://wandb.ai/
WANDB_API_KEY=NotSpecified

# GITHUB configuration only required for building the image.
GITHUB_ACCESS_TOKEN=INSERT_GITHUB_ACCESS_TOKEN_HERE
GITHUB_BRANCH=main
```

### Build Container

**Use of the provided container from NGC is *highly* recommended. However, it is possible to build a container from scratch for customization.**

The `launch.sh` script can be used to build and push containers to a registry. It can also be used to run interactive development jobs on a local system. See the instructions inside the script for more information. Once the `.env` script is created, a container can be built by running `bash launch.sh build`. If pushing to a registry is desired, `bash launch.sh push` will complete this task.

### Setup Data Processing and Training Files

The following elements are required to process data or run pre-training.

#### Model Configuration File

MegaMolBART uses yaml based parameter files. The existing model configuration files can be found in `examples/chem/conf` and are based on the configurations for [Chemformer](https://chemrxiv.org/engage/chemrxiv/article-details/60ee8a3eb95bdd06d062074b):

* [`megamolbart_pretrain_small_span_aug.yaml`](./examples/chem/conf/megamolbart_pretrain_small_span_aug.yaml): Small model with augmentation of the encoder SMILES and masking of decoder tokens
* [`megamolbart_pretrain_large_span_aug.yaml`](./examples/chem/conf/megamolbart_pretrain_large_span_aug.yaml): Large model with augmentation of the encoder SMILES and masking of decoder tokens
* [`megamolbart_pretrain_xsmall_span_aug.yaml`](./examples/chem/conf/megamolbart_pretrain_xsmall_span_aug.yaml): Extra small model with augmentation of the encoder SMILES and masking of decoder tokens, mainly for testing and is used by default.

Additional files can be created to suit other configurations. Though data processing does not require an explicit model configuration, one of these files (or the default) must be provided.

#### Python Run Script

The yaml parameter files are read by the python file [`megamolbart_pretrain.py`](./examples/chem/megamolbart_pretrain.py) which runs the data processing and/or training loop. Typically changes will not need to be made to this file.

#### Shell Execution Script

The bash scripts found in [`examples/chem/slurm`](./examples/chem/slurm) and [`examples/chem/shell`](./examples/chem/shell), respectively, are configurable for the location of system directories for data, results, and (optionally) development code. These files are also used to configure the size of the training run (number of nodes and gpus). Note that multi-node training has been validated on SLURM-based systems.

### Data Preprocessing

#### Download and Pre-Process CSV Data

By default, the python pretrain script uses a subset of the selected ZINC15 tranches to process and train. This is to prevent ~100GB of data from being downloaded accidentally. To process with all of the data from the ZINC15 tranches, the following modification is required:

> Change the `links_file` path from `'conf/dataset/ZINC-downloader-test.txt'` to `'conf/dataset/ZINC-downloader.txt'` in the config file [here](https://github.com/clara-parabricks/NeMo_MegaMolBART/blob/dev/examples/chem/conf/megamolbart_pretrain_base.yaml)

```yaml
links_file: 'conf/dataset/ZINC-downloader.txt' # to process with all of the ZINC15 data
```

The data processing can then be run as described in the [Data Processing section of the Quickstart Guide](./QUICKSTART.md#data-processing). Alternatively, to avoid having to set `++do_training=False` from the command line, ensure that the following is set (or added) to the top level of the appropriate yaml config file:

```yaml
do_training: False # set to false if data preprocessing steps must be completed
```

#### Download and Pre-Process Binary Data (BETA)

Training can also be performed using NeMo's Megatron binary data format, which in some cases might further accelerate training times relative to CSV. This data format will first create a CSV version of the data (see above section) if it does not already exist. If a CSV version is already present, it will be used.

To create binary data, ensure the following is added to the `model` section of the yaml file, during data processing. The binary data format pre-calculates a user-specified maximum number of enumerated SMILES molecules. This can be specified, and a suggested default is 5. Setting the number of enumerations to a very large number will result in larger data files and longer data pre-processing times.

```yaml
model:
  data:
    dataset_format: bin # Set to csv (default) or bin
    num_enumerations: 5 # Set to the number of SMILES enumerations to be stored
```

### Pre-Training

Training is performed as described in the [Training MegaMolBART section of the Quickstart Guide](./QUICKSTART.md#training-megamolbart). One additional feature of note is that ranges of data files can be selected. For example, to select all of the data files (numbered 000 to 146), use the range indicator `x[000..146]`. For only ten files, use `x[000..009]`. Ensure these are set as appropriate for the train, validation, and test splits as below in the yaml config file:

```yaml
model:
  data:
    dataset_format: csv # Set to csv (default) or bin
    dataset:
      train: x[000..146]
      test: x[000..146]
      val: x[000..146]
```

**NOTE**: Do NOT add an extension to the data file lists. The appropriate extension (csv or bin) is added automatically.

The training can be run as described in the [Quickstart Guide](./QUICKSTART.md#training-megamolbart). Additional parameters can be set in the yaml configuration file or in the shell/SLURM script as appropriate.


### Testing model
Please follow these instructions to inference a trained model.

1. Add 'MODEL_PATH' in .env file located in the source code root dir.
   Please ensure the value is the directory on the host machine containing the model file.
2. Edit 'examples/chem/conf/infer.yaml' to update 'model.model_path'.
   Change the filename to the desired model name. Please donot change the folder name.
   e.g.
    ```
    model:
	    model_path: /models/<<CHANGE ME.nemo>>
    ```
3. Execute './launch dev'

To test/verify all inference operations please execute `pytests --log-cli-level=10 tests/test_inference.py` inside the container.

## References

1. Sterling T., and Irwin, J., *Chem. Inf. Model*, 2015, [doi](http://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559).
2. Irwin R., Dimitriadis S., He J., and Bjerrum E., "Chemformer: A Pre-Trained Transformer for Computational Chemistry", *Mach. Learn.: Sci. Technol.*, 2022, [doi](https://iopscience.iop.org/article/10.1088/2632-2153/ac3ffb).
