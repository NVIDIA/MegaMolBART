# Copyright (c) 2022, NVIDIA CORPORATION.
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

# TODO: Fix the image for GA
FROM nvcr.io/nv-drug-discovery-dev/nemo:1.11rc-mmb
# FROM gitlab-master.nvidia.com/mlivne/nemo_containers:nemo-r1.11.0

ARG GITHUB_ACCESS_TOKEN
ARG GITHUB_BRANCH
ARG NEMO_BRANCH
ARG NEMO_CHEM_HOME=/opt/nvidia/nemo_chem
ARG NEMO_HOME=/opt/nvidia/nemo
ARG CONDA_ENV=base
ARG PACKAGE=0

# Activating base environment
RUN echo "source activate ${CONDA_ENV}" > /etc/bash.bashrc
ENV TERM=xterm

## Installing dependencies and tools
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        python-setuptools python-dev swig \
        wget git unzip tmux vim && \
    rm -rf /var/lib/apt/lists/*

## Setup - Install and upgrade additional packages
# RUN pip install --upgrade numpy wandb
# RUN pip install --upgrade einops
COPY setup/conda/env.yml /tmp/env.yml
RUN conda env update --name base -f /tmp/env.yml && conda clean -afy

## PySMILES
RUN git clone https://github.com/MolecularAI/pysmilesutils.git --branch master /opt/pysmilesutils \
    && cd /opt/pysmilesutils; pip install .

RUN git clone --branch ${GITHUB_BRANCH} \
    https://${GITHUB_ACCESS_TOKEN}@github.com/clara-parabricks/NeMo_MegaMolBART.git ${NEMO_CHEM_HOME}

RUN cd ${NEMO_CHEM_HOME} && conda run -n ${CONDA_ENV} pip install -e .

# Recompile Megatron helpers -- NOTE this will need to be done if PYTHONPATH changes later
ENV PYTHONPATH=${NEMO_CHEM_HOME}:${NEMO_CHEM_HOME}/generated:${NEMO_HOME}:$PYTHONPATH
ENV PACKAGE=${PACKAGE}
ENV NEMO_CHEM_HOME=${NEMO_CHEM_HOME}
ENV NEMO_HOME=${NEMO_HOME}
ENV NEMO_BRANCH=${NEMO_BRANCH}

RUN NEMO_PATH=$(python -c 'import nemo; print(nemo.__path__[0])') && \
   cd ${NEMO_PATH}/collections/nlp/data/language_modeling/megatron && \
   make

COPY ./.tmp/ /tmp/
COPY ./setup/config_model.sh /tmp
COPY ./setup/startup.sh /opt/nvidia
COPY ./setup/entrypoint.sh /opt
RUN chmod +x /opt/entrypoint.sh
RUN bash /tmp/config_model.sh ${PACKAGE}
WORKDIR ${NEMO_CHEM_HOME}
COPY ./.tmp/ /tmp/

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD [ "bash", "-c", "/opt/nvidia/startup.sh"]
