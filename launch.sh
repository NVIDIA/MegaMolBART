#!/bin/bash
#
# Copyright (c) 2020, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###############################################################################
#
# This is my $LOCAL_ENV file
#
LOCAL_ENV=.env
#
###############################################################################

usage() {
    cat <<EOF

USAGE: launch.sh

launch utility script
----------------------------------------

launch.sh [command]

    valid commands:

    build
    pull
    push
    dev
    root
    jupyter


Getting Started tl;dr
----------------------------------------

    ./launch.sh build
    ./launch.sh dev
For more detailed info on getting started, see README.md


More Information
----------------------------------------

Note: This script looks for a file called $LOCAL_ENV in the
current directory. This file should define the following environment
variables:
    MEGAMOLBART_CONT
        container image for MegaMolBART training, prepended with registry. e.g.,
        Note that this is a separate (precursor) container from any service associated containers
    PROJECT_PATH
        local path to code. e.g., /home/user/code/MegaMolBART
    JUPYTER_PORT
        Port for launching jupyter lab, e.g. 8888
    DATA_PATH
        path to data directory. e.g., /scratch/data/zinc_csv_split
    REGISTRY
        container registry URL. e.g., nvcr.io. Only required to push/pull containers.
    REGISTRY_USER
        container registry username. e.g., '$oauthtoken' for registry access. Only required to push/pull containers.
    REGISTRY_ACCESS_TOKEN
        container registry access token. e.g., Ckj53jGK... Only required to push/pull containers.
    WANDB_API_KEY
        Weights and Balances API key to upload runs to WandB. Can also be uploaded afterwards., e.g. Dkjdf...
        This value is optional -- Weights and Biases will log data and not upload if missing.
    GITHUB_ACCESS_TOKEN
        GitHub API token to checkout private code repo (required for build only)

EOF
    exit
}

MEGAMOLBART_CONT=${MEGAMOLBART_CONT:=nvcr.io/nvidia/clara/megamolbart:0.2.0}
PROJECT_PATH=${PROJECT_PATH:=$(pwd)}
DATA_PATH=${DATA_PATH:=/tmp}
RESULT_PATH=${RESULT_PATH:=${HOME}/results/nemo_experiments}

JUPYTER_PORT=${JUPYTER_PORT:=8888}

REGISTRY_USER=${REGISTRY_USER:='$oauthtoken'}
REGISTRY=${REGISTRY:=NotSpecified}
REGISTRY_ACCESS_TOKEN=${REGISTRY_ACCESS_TOKEN:=NotSpecified}

WANDB_API_KEY=${WANDB_API_KEY:=NotSpecified}

GITHUB_ACCESS_TOKEN=${GITHUB_ACCESS_TOKEN:=UserName:PersonalToken}
GITHUB_BRANCH=${GITHUB_BRANCH:=main}

# if $LOCAL_ENV file exists, source it to specify my environment
if [ -e ./$LOCAL_ENV ]
then
    echo sourcing environment from ./$LOCAL_ENV
    . ./$LOCAL_ENV
    write_env=0
else
    echo $LOCAL_ENV does not exist. Writing deafults to $LOCAL_ENV
    write_env=1
fi

# If $LOCAL_ENV was not found, write out a template for user to edit
if [ $write_env -eq 1 ]; then
    echo MEGAMOLBART_CONT=${MEGAMOLBART_CONT} >> $LOCAL_ENV
    echo PROJECT_PATH=${PROJECT_PATH} >> $LOCAL_ENV
    echo DATA_PATH=${DATA_PATH} >> $LOCAL_ENV
    echo RESULT_PATH=${RESULT_PATH} >> $LOCAL_ENV
    echo JUPYTER_PORT=${JUPYTER_PORT} >> $LOCAL_ENV
    echo REGISTRY_USER=${REGISTRY_USER} >> $LOCAL_ENV
    echo REGISTRY=${REGISTRY} >> $LOCAL_ENV
    echo REGISTRY_ACCESS_TOKEN=${REGISTRY_ACCESS_TOKEN} >> $LOCAL_ENV
    echo GITHUB_ACCESS_TOKEN=${GITHUB_ACCESS_TOKEN} >> $LOCAL_ENV
    echo WANDB_API_KEY=${WANDB_API_KEY} >> $LOCAL_ENV
    echo GITHUB_BRANCH=${GITHUB_BRANCH} >> $LOCAL_ENV
fi

PROJECT_MOUNT_PATH="/workspace/nemo_chem"
DATA_MOUNT_PATH="/data"
RESULT_MOUNT_PATH='/result/nemo_experiments'
DEV_CONT_NAME='nemo_megamolbart'

# Additional variables when send in .env file, is used in the script:
# BASE_IMAGE        Custom Base image for building.
# NEMO_PATH         Path to NeMo source cdoe.
# CHEM_BENCH_PATH   Path to chembench source code. Used for generating benchmark
#                   data
# MODEL_PATH        Local dir to be mounted to /model
# MODEL_FILE        Model file in MODEL_PATH to be packaged duing build

# Compare Docker version to find Nvidia Container Toolkit support.
# Please refer https://github.com/NVIDIA/nvidia-docker
DOCKER_VERSION_WITH_GPU_SUPPORT="19.03.0"
if [ -x "$(command -v docker)" ]; then
    DOCKER_VERSION=$(docker version | grep -i version | head -1 | awk '{print $2'})
fi

PARAM_RUNTIME="--runtime=nvidia"
if [ "$DOCKER_VERSION_WITH_GPU_SUPPORT" == "$(echo -e "$DOCKER_VERSION\n$DOCKER_VERSION_WITH_GPU_SUPPORT" | sort -V | head -1)" ];
then
    PARAM_RUNTIME="--gpus all"
fi

DOCKER_CMD="docker run \
    --network host \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v /etc/shadow:/etc/shadow:ro \
    -u $(id -u):$(id -u) \
    ${PARAM_RUNTIME} \
    -p ${JUPYTER_PORT}:8888 \
    -v ${PROJECT_PATH}:${PROJECT_MOUNT_PATH} \
    -v ${DATA_PATH}:${DATA_MOUNT_PATH} \
    -v ${RESULT_PATH}:${RESULT_MOUNT_PATH}
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e HOME=${PROJECT_MOUNT_PATH} \
    -e TMPDIR=/tmp/ \
    -e NUMBA_CACHE_DIR=/tmp/ \
    -w ${PROJECT_MOUNT_PATH} "

DOCKER_BUILD_CMD="docker build --network host \
    -t ${MEGAMOLBART_CONT} \
    --build-arg GITHUB_ACCESS_TOKEN=${GITHUB_ACCESS_TOKEN} \
    --build-arg GITHUB_BRANCH=${GITHUB_BRANCH} \
    --build-arg NEMO_MEGAMOLBART_HOME=${PROJECT_MOUNT_PATH} \
    --build-arg NEMO_BRANCH=${NEMO_BRANCH} \
    -f setup/Dockerfile"

build() {
    local IMG_NAME=($(echo ${MEGAMOLBART_CONT} | tr ":" "\n"))
    local PACKAGE=0

    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--pkg)
                PACKAGE=1
                shift
                ;;
            -b|--base-image)
                BASE_IMAGE=$2
                shift
                shift
                ;;
            -c|--clean)
                DOCKER_BUILD_CMD="${DOCKER_BUILD_CMD} --no-cache"
                shift
                ;;
            *)
                echo "Unknown option $1. Please --version to specify a version."
                exit 1
                ;;
        esac
    done

    if [ ${PACKAGE} -eq 1 ]
    then
        set -e
        local MODEL_FILE="${MODEL_PATH}/${MODEL_FILE}"
        echo "Coping model from ${MODEL_FILE}..."
        rm -rf ./.tmp/
        mkdir -p ./.tmp/models
        cp ${MODEL_FILE} ./.tmp/models
        set +e
    else
        mkdir -p ./.tmp/models
    fi

    if [ ! -z "${BASE_IMAGE}" ];
    then
        DOCKER_BUILD_CMD="${DOCKER_BUILD_CMD} --build-arg BASE_IMAGE=${BASE_IMAGE}"
    fi

    DOCKER_BUILD_CMD="${DOCKER_BUILD_CMD} --build-arg PACKAGE=${PACKAGE}"
    DOCKER_BUILD_CMD="${DOCKER_BUILD_CMD} -t ${IMG_NAME[0]}:latest"

    echo "Building MegaMolBART training container..."
    set -x
    ${DOCKER_BUILD_CMD} .
    set +x
    exit
}


push() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--version)
                VERSION="$2"
                shift
                shift
                ;;
            -a|--additional_copies)
                ADDITIONAL_IMAGES="$2"
                shift
                shift
                ;;
            *)
                echo "Unknown option $1. Please --version to specify a version."
                exit 1
                ;;
        esac
    done

    local IMG_NAME=($(echo ${MEGAMOLBART_CONT} | tr ":" "\n"))

    docker login ${REGISTRY} -u ${REGISTRY_USER} -p ${REGISTRY_ACCESS_TOKEN}
    docker push ${IMG_NAME[0]}:latest
    docker push ${MEGAMOLBART_CONT}

    if [ ! -z "${VERSION}" ];
    then
        docker tag ${MEGAMOLBART_CONT} ${IMG_NAME[0]}:${VERSION}
        docker push ${IMG_NAME[0]}:${VERSION}
    fi

    if [ ! -z "${ADDITIONAL_IMAGES}" ];
    then
        IFS=',' read -ra IMAGES <<< ${ADDITIONAL_IMAGES}
        for IMAGE in "${IMAGES[@]}"; do
            docker tag ${MEGAMOLBART_CONT} ${IMAGE}
            docker push ${IMAGE}
        done
    fi

    exit
}


setup() {
    mkdir -p ${DATA_PATH}
    mkdir -p ${RESULT_PATH}

    DEV_PYTHONPATH="/workspace/nemo_chem:/workspace/nemo_chem/generated"
    if [ ! -z "${NEMO_PATH}" ];
    then
        DOCKER_CMD="${DOCKER_CMD} -v ${NEMO_PATH}:/workspace/nemo "
        DEV_PYTHONPATH="${DEV_PYTHONPATH}:/workspace/nemo"
    fi

    if [ ! -z "${CHEM_BENCH_PATH}" ];
    then
        DOCKER_CMD="${DOCKER_CMD} -v ${CHEM_BENCH_PATH}:/workspace/chembench "
        DEV_PYTHONPATH="${DEV_PYTHONPATH}:/workspace/chembench"
    fi

    if [ ! -z "${MODEL_PATH}" ];
    then
        DOCKER_CMD="${DOCKER_CMD} -v ${MODEL_PATH}:/models"
    fi

    DOCKER_CMD="${DOCKER_CMD} --env PYTHONPATH=${DEV_PYTHONPATH}"
    DOCKER_CMD="${DOCKER_CMD} --env WANDB_API_KEY=$WANDB_API_KEY"
}


dev() {
    CMD='bash'
    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--additional-args)
                DOCKER_CMD="${DOCKER_CMD} $2"
                shift
                shift
                ;;
            -t|--tmp)
                DEV_CONT_NAME="${DEV_CONT_NAME}_$2"
                shift
                shift
                ;;
            -d|--demon)
                DOCKER_CMD="${DOCKER_CMD} -d"
                shift
                ;;
            -n|--notebook_home)
                DOCKER_CMD="${DOCKER_CMD} -e NOTEBOOK_HOME=$2"
                shift
                shift
                ;;
            -c|--cmd)
                shift
                CMD="$@"
                break
                ;;
            *)
                echo "Unknown option '$1'.
Available options are -a(--additional-args), -i(--image), -d(--demon) and -c(--cmd)"
                exit 1
                ;;
        esac
    done

    setup
    set -x
    ${DOCKER_CMD} --rm -it --name ${DEV_CONT_NAME} ${MEGAMOLBART_CONT} ${CMD}
    set +x
    exit
}


run() {
    setup
    set -x
    ${DOCKER_CMD} -d ${MEGAMOLBART_CONT} ${@:1}
    set +x
    exit
}


attach() {
    set -x
    DOCKER_CMD="docker exec"
    CONTAINER_ID=$(docker ps | grep ${DEV_CONT_NAME} | cut -d' ' -f1)
    ${DOCKER_CMD} -it ${CONTAINER_ID} /bin/bash
    exit
}


jupyter() {
    setup
    ${DOCKER_CMD} -it ${MEGAMOLBART_CONT} jupyter-lab --no-browser \
        --port=${JUPYTER_PORT} \
        --ip=0.0.0.0 \
        --allow-root \
        --notebook-dir=/workspace \
        --NotebookApp.password='' \
        --NotebookApp.token='' \
        --NotebookApp.password_required=False
}


case $1 in
    build)
        $@
        ;;
    push)
        $@
        ;;
    dev)
        $@
        ;;
    run)
        $@
        ;;
    attach)
        $@
        ;;
    jupyter)
        $1
        exit 0
        ;;
    *)
        usage
        ;;
esac
