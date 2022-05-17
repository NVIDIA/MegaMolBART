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

from typing import List
from enum import Enum
import re
import braceexpand
import os
from copy import deepcopy
from omegaconf import DictConfig, open_dict
import torch.utils.data as pt_data
from pytorch_lightning.trainer.trainer import Trainer

from nemo.utils import logging
from .csv_dataset import MoleculeCsvDataset
from .molecule_binary_dataset import MoleculeBinaryDataset

__all__ = ['DatasetTypes', 'expand_dataset_paths', 'build_train_valid_test_datasets']

class DatasetTypes(Enum):
    zinc_csv  = 0


def expand_dataset_paths(filepath: str, ext: str) -> List[str]:
    """Expand dataset paths from braces"""
    filepath = filepath + ext if ext else filepath
    # TODO this should eventually be moved to a Nemo fileutils module or similar
    filepath = re.sub(r"""\(|\[|\<|_OP_""", '{', filepath) # replaces '(', '[', '<' and '_OP_' with '{'
    filepath = re.sub(r"""\)|\]|\>|_CL_""", '}', filepath) # replaces ')', ']', '>' and '_CL_' with '}'
    dataset_paths = list(braceexpand.braceexpand(filepath))
    return dataset_paths


def _build_train_valid_test_datasets(
    cfg: DictConfig,
    trainer: Trainer,
    num_samples: int,
    filepath: str,
    metadata_path: str,
    dataset_format: str
):
    # TODO num_samples is currently not used

    cfg = deepcopy(cfg)
    with open_dict(cfg):
        cfg['metadata_path'] = metadata_path

    # Get datasets and load data
    logging.info(f'Loading data from {filepath}')
    dataset_paths = expand_dataset_paths(filepath, ".csv") if dataset_format == "csv" else expand_dataset_paths(filepath, None)
    logging.info(f'Loading data from {dataset_paths}')
    dataset_list = []
    if dataset_format == "csv":
        dataset = MoleculeCsvDataset(dataset_paths=dataset_paths, cfg=cfg, trainer=trainer)
    elif dataset_format == "bin":
        for path in dataset_paths:
            data = MoleculeBinaryDataset(filepath=path, cfg=cfg, trainer=trainer, num_samples=num_samples)
            dataset_list.append(data)
            num_samples -= len(data)
            if num_samples < 1:
                break
        if len(dataset_list) == 1:
            dataset = dataset_list[0]
        else:
            dataset = pt_data.ConcatDataset(dataset_list)
    else:
        raise ValueError("Unrecognized data format. Expected csv or bin.")
    return dataset


def build_train_valid_test_datasets(
    cfg: DictConfig,
    trainer: Trainer,
    train_valid_test_num_samples: List[int]
):
     # TODO metadata_file is currently not used
     
    cfg = deepcopy(cfg)
    with open_dict(cfg):
        dataset_path = cfg.pop('dataset_path', '')
        dataset_files = cfg.pop('dataset_files')
        metadata_file = cfg.pop('metadata_file', None)
        dataset_format = cfg.pop('dataset_format')

    # Build individual datasets.
    filepath = os.path.join(dataset_path, 'train', dataset_files)
    metadata_path = os.path.join(dataset_path, 'train', metadata_file) if metadata_file else None
    train_dataset = _build_train_valid_test_datasets(cfg, trainer, train_valid_test_num_samples[0],
                                                     filepath, metadata_path, dataset_format)

    filepath = os.path.join(dataset_path, 'val', dataset_files)
    metadata_path = os.path.join(dataset_path, 'val', metadata_file) if metadata_file else None
    validation_dataset = _build_train_valid_test_datasets(cfg, trainer, train_valid_test_num_samples[1],
                                                          filepath, metadata_path, dataset_format)

    filepath = os.path.join(dataset_path, 'test', dataset_files)
    metadata_path = os.path.join(dataset_path, 'test', metadata_file) if metadata_file else None
    test_dataset = _build_train_valid_test_datasets(cfg, trainer, train_valid_test_num_samples[2],
                                                    filepath, metadata_path, dataset_format)

    return (train_dataset, validation_dataset, test_dataset)
