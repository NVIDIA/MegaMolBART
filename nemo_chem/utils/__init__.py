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

from pathlib import Path
from collections import defaultdict
from dataclasses import asdict

from nemo.utils import logging


def update_dataclass_config(cfg, dataset_config_class):
    """Update a dataset configuration with existing defaults"""
    default_cfg = asdict(dataset_config_class())
    default_cfg.update(cfg)
    return default_cfg


def recursive_make_dirs(directory):
    """Recursively create required directory structure"""
    logging.info(f'Creating directory {str(directory)}...')
    if isinstance(directory, str):
        directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)


def flatten_dict(list_of_dicts):
    """Flatten a list of dictionaries to list without assuming all keys are identical"""
    flattened_dict = defaultdict(list)
    for metric_dict in list_of_dicts:
        for metric_name, metric_value in metric_dict.items():
            flattened_dict[metric_name].append(metric_value)

    return flattened_dict
