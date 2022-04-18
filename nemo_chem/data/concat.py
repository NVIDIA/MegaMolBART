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

from typing import Any, List, Iterable

import numpy as np
import torch.utils.data as pt_data
from nemo.core import IterableDataset
from nemo.core import Dataset
from nemo.utils import logging
import itertools


class ConcatIterableDataset(IterableDataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets. It is identical
    to PyTorch's version except it allows iterable datasets to be use if they have 
    a known length.

    Args:
        datasets (sequence): List of datasets to be concatenated

    
    """
    datasets: List[IterableDataset]
    cumulative_sizes: List[int]

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        # TODO remove before v0.2 release
        logging.warning("ConcatIterableDataset is not compatible with NeMo's Megatron dataloaders and is deprecated")
        super(ConcatIterableDataset, self).__init__()
        assert len(datasets) > 0, AssertionError('Datasets should not be an empty iterable.')
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __iter__(self):
        while True:
            for _ in range(self.cumulative_sizes[-1]):
                yield next(itertools.chain.from_iterable(self.datasets)) # TODO ensure this stores state