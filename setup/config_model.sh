#!/bin/bash
set -e
set -x

PACKAGE=$1

if [ ${PACKAGE} -eq 1 ]
then
    echo "Coping model from /tmp/models/ ..."
    mv /tmp/models/ /
    ls /models/
fi
