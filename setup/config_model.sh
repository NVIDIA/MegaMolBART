#!/bin/bash
set -e
set -x

PACKAGE=$1

if [ ${PACKAGE} -eq 1 ]
then
    echo "Coping model from /tmp/models/ ..."
    mv /tmp/models/ /
    rm -rf ${NEMO_HOME}
    mkdir -p /opt/nvidia
    git clone -b ${NEMO_BRANCH} https://github.com/michalivne/NeMo.git ${NEMO_HOME}
    ls ${NEMO_HOME}
    ls /models/
fi

#rm /tmp/models
