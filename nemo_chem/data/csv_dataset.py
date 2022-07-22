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

import os
import time
import pickle

from typing import Optional
from dataclasses import dataclass

import torch
import numpy as np

from nemo.core import Dataset
from nemo.utils import logging
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import CSVMemMapDataset

try:
    from apex.transformer.parallel_state import get_rank_info
    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

__all__ = ['MoleculeCsvDatasetConfig', 'MoleculeCsvDataset', 'DatasetFileConfig']

@dataclass
class DatasetFileConfig():
    train: str = None
    test: str = None
    val: str = None

@dataclass
class MoleculeCsvDatasetConfig():
    dataset_path: str = ''
    dataset: DatasetFileConfig = None
    newline_int: int = 10
    header_lines: int = 1
    data_col: int = 1
    data_sep: str = ','
    sort_dataset_paths: bool = True
    # FIXME: remove unneeded config variables
    skip_lines: int = 0
    micro_batch_size: int = 1
    encoder_augment: bool = False
    encoder_mask: bool = False
    decoder_augment: bool = False
    decoder_mask: bool = False
    canonicalize_input: bool = True
    dataloader_type: str = 'single'
    drop_last: bool = False
    pin_memory: bool = False # must be False with CSV dataset
    num_workers: Optional[int] = None


class MoleculeCsvDataset(CSVMemMapDataset):
    """
    Allow per-line lazy access to multiple text files using numpy memmap.
    """
    def __init__(self,
                 dataset_paths,
                 cfg,
                 workers=None):
        super().__init__(
            dataset_paths=dataset_paths,
            newline_int=cfg.get('newline_int'),
            header_lines=cfg.get('header_lines'), # skip first N lines
            workers=workers,
            tokenizer=None,
            sort_dataset_paths=cfg.get('sort_dataset_paths'),
            data_col=cfg.get('data_col'),
            data_sep=cfg.get('data_sep'),
        )
