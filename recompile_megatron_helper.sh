#!/bin/bash

# Find NeMo installation location and re-combile Megatron helpers
NEMO_PATH=$(python -c 'import nemo; print(nemo.__path__[0])')
cd ${NEMO_PATH}/collections/nlp/data/language_modeling/megatron
make

