#!/bin/bash
#
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

# Process ZINC CSV data
# This script processes the ZINC dataset used to train MolBART
# so that train, val, and test splits are in separate directories
# It also removes the set column since it is no longer needed and
# will create a metadata file containing the number of molecules
# in each file

# Example input data format:
# zinc_id,smiles,set
# ZINC000843130676,CCN1CCN(c2ccc(-c3nc(CCN)no3)cc2F)CC1,train
# ZINC000171110690,CC(C)(C)c1noc(CSCC(=O)NC2CC2)n1,train
# ZINC000848409174,CC(=NN[C@H]1CCCOC1)c1cncnc1C,train

SOURCE_DIR=./zinc_csv # location of original data
DEST_DIR=./zinc_csv_split # location of new data


######


METADATA_FILE=metadata.txt

for SPLIT in "train" "val" "test"; do
    echo "Processing $SPLIT.."
    mkdir -p $DEST_DIR/${SPLIT}
    echo "file,size" > ${DEST_DIR}/${SPLIT}/${METADATA_FILE} # metadata file with number of molecules

    for f in $SOURCE_DIR/*.csv; do
        echo "Processing $f file.."

        BASE_FILENAME=`basename $f`
        DEST_FILE=${DEST_DIR}/${SPLIT}/${BASE_FILENAME}

        # Destination data
        echo "zinc_id,smiles" > ${DEST_FILE} # file header
        cat $f | grep $SPLIT | cut -d',' -f1,2 >> ${DEST_FILE} # output entries for split

        # Log number of molecules
        NUM_MOL=$(wc -l ${DEST_FILE}| cut -d' ' -f1)
        NUM_MOL=$(($NUM_MOL-1))
        echo "$BASE_FILENAME,$NUM_MOL" >> ${DEST_DIR}/${SPLIT}/${METADATA_FILE}
    done
done


