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
LOCAL_ENV=.cheminf_local_environment
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
    root
    jupyter


Getting Started tl;dr
----------------------------------------

    ./launch build
    ./launch dev
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
        path to repository. e.g.,
        /home/user/projects/cheminformatics
    REGISTRY_ACCESS_TOKEN
        container registry access token. e.g.,
        Ckj53jGK...
    REGISTRY_USER
        container registry username. e.g.,
        astern
    REGISTRY
        container registry URL. e.g.,
        server.com/registry:5005
    DATA_PATH
        path to data directory. e.g.,
        /scratch/data/cheminformatics
    JUPYTER_PORT
        Port for launching jupyter lab
    GITHUB_ACCESS_TOKEN
        GitHub API token for repo access

EOF
    exit
}


###############################################################################
#
# if $LOCAL_ENV file exists, source it to specify my environment
#
###############################################################################

if [ -e ./$LOCAL_ENV ]
then
    echo sourcing environment from ./$LOCAL_ENV
    . ./$LOCAL_ENV
    write_env=0
else
    echo $LOCAL_ENV does not exist. Writing deafults to $LOCAL_ENV
    write_env=1
fi

###############################################################################
#
# alternatively, override variable here.  These should be all that are needed.
#
###############################################################################

MEGAMOLBART_CONT=${MEGAMOLBART_CONT:=nvcr.io/nvidian/clara-lifesciences/megamolbart_training}
PROJECT_PATH=${PROJECT_PATH:=$(pwd)}
JUPYTER_PORT=${JUPYTER_PORT:-9000}
DATA_PATH=${DATA_PATH:=/tmp}
DATA_MOUNT_PATH=${DATA_MOUNT_PATH:=/data}
RESULT_MOUNT_PATH=${RESULT_MOUNT_PATH:=/result/nemo_experiments}
RESULT_PATH=${RESULT_PATH:=/home/mgill/results/nemo_experiments}

###############################################################################
#
# If $LOCAL_ENV was not found, write out a template for user to edit
#
###############################################################################

if [ $write_env -eq 1 ]; then
    echo MEGAMOLBART_CONT=${MEGAMOLBART_CONT} >> $LOCAL_ENV
    echo PROJECT_PATH=${PROJECT_PATH} >> $LOCAL_ENV
    echo JUPYTER_PORT=${JUPYTER_PORT} >> $LOCAL_ENV
    echo DATA_PATH=${DATA_PATH} >> $LOCAL_ENV
    echo DATA_MOUNT_PATH=${DATA_MOUNT_PATH} >> $LOCAL_ENV
fi

###############################################################################
#
#          shouldn't need to make changes beyond this point
#
###############################################################################
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

NEMO_HOME=/workspace/nemo
DOCKER_CMD="docker run \
    --rm \
    --network host \
    ${PARAM_RUNTIME} \
    -p ${JUPYTER_PORT}:8888 \
    -v ${PROJECT_PATH}:${NEMO_HOME} \
    -v ${DATA_PATH}:${DATA_MOUNT_PATH} \
    -v /home/mgill/.ssh:${NEMO_HOME}/.ssh:ro \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e HOME=${NEMO_HOME} \
    -e TF_CPP_MIN_LOG_LEVEL=3 \
    -w ${NEMO_HOME}"
 
DATE=$(date +%y%m%d)

build() {
    set -e
    MEGAMOLBART_CONT_BASENAME="$( cut -d ':' -f 1 <<< "$MEGAMOLBART_CONT" )"
    echo "Building MegaMolBART training container..."
    docker build --network host \
        -t ${MEGAMOLBART_CONT_BASENAME}:latest \
        -t ${MEGAMOLBART_CONT_BASENAME}:${DATE} \
        --build-arg GITHUB_ACCESS_TOKEN=${GITHUB_ACCESS_TOKEN} \
        -f Dockerfile.nemo_chem \
        .

    set +e
    exit
}


push() {
    docker login ${REGISTRY} -u ${REGISTRY_USER} -p ${REGISTRY_ACCESS_TOKEN}
    docker push ${MEGAMOLBART_CONT}:latest
    docker push ${MEGAMOLBART_CONT}:${DATE}
    exit
}


pull() {
    docker login ${REGISTRY} -u ${REGISTRY_USER} -p ${REGISTRY_ACCESS_TOKEN}
    docker pull ${MEGAMOLBART_CONT}
    exit
}


dev() {
    set -x
    DOCKER_CMD="${DOCKER_CMD} -v ${RESULT_PATH}:${RESULT_MOUNT_PATH} --name nemo_dev " 
    ${DOCKER_CMD} -it ${MEGAMOLBART_CONT} bash
    exit
}


root() {
    ${DOCKER_CMD} -it --user root ${MEGAMOLBART_CONT} bash
    exit
}


jupyter() {
    ${DOCKER_CMD} -it ${MEGAMOLBART_CONT} jupyter-lab --no-browser \
        --port=8888 \
        --ip=0.0.0.0 \
        --notebook-dir=/workspace \
        --NotebookApp.password=\"\" \
        --NotebookApp.token=\"\" \
        --NotebookApp.password_required=False
    exit
}


case $1 in
    build)
        ;&
    push)
        ;&
    pull)
        ;&
    dev)
        $@
        ;;
    root)
        ;&
    jupyter)
        $1
        ;;
    *)
        usage
        ;;
esac
