#!/bin/bash

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

# Usage:
# ./auto_launcher.sh -n 5 <file.sub>
# Grab command line options
# n: Number of times to submit the job

N_CALLS=1

while getopts "n:J:" opt; do
  case $opt in
    n) N_CALLS=$OPTARG;;
  esac
done

# Grab the .sub file to run
SUBFILE=${@:$OPTIND:1}
if [[ -z $SUBFILE ]]; then
  echo "Usage: $(basename "$0") [flags] [sub file]"
  exit 1
fi
echo "Calling [$SUBFILE] $N_CALLS times."

# Repeat calls
PREV_JOBID=""
for (( i = 1; i <= $N_CALLS; i++ ))
do
  if [ -z $PREV_JOBID ]; then
    echo "Submitting job ${i}"
    OUTPUT=$(sbatch $SUBFILE)
  else
    echo "Submitting job ${i} w/ dependency on jobid ${PREV_JOBID}"
    OUTPUT=$(sbatch --dependency=afterany:${PREV_JOBID} $SUBFILE)
  fi
  PREV_JOBID="$(cut -d' ' -f4 <<< $OUTPUT)"
done
