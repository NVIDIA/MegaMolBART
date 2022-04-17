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
import re
import math
import mmap
from typing import Optional
from dataclasses import dataclass

import torch
from nemo.core import Dataset, IterableDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_dataset import MegatronDataset
from nemo.utils import logging

__all__ = ['MoleculeCsvDatasetConfig', 'MoleculeCsvDataset', 'MoleculeCsvIterableDataset']


@dataclass
class MoleculeCsvDatasetConfig():
    dataset_path: str = ''
    dataset_files: str = 'data.csv'
    metadata_file: str = 'metadata.txt'
    dataset_type: str = 'zinc_csv'
    micro_batch_size: int = 1
    use_iterable: bool = False
    map_data: bool = False
    encoder_augment: bool = True
    encoder_mask: bool = False
    decoder_augment: bool = False
    canonicalize_input: bool = True # TODO remove when CSV data processing updated
    drop_last: bool = False
    shuffle: bool = False
    num_workers: Optional[int] = None
    pin_memory: bool = True # TODO: remove this if value is fixed
    dataloader_type: str = 'single'


class MoleculeCsvABCDataset(MegatronDataset):
    """Molecule base dataset that reads SMILES from the second column from CSV files."""
    def __init__(self, filepath, cfg, trainer, num_samples=None):
        """
        Args:
            dataset_cfg: dataset config
            trainer: Pytorch Lightning trainer
        """
        self.cfg = cfg
        assert os.path.exists(filepath), FileNotFoundError(f"Could not find CSV file {filepath}")
        super().__init__(cfg=self.cfg, trainer=trainer)

        self.filepath = filepath
        self.map_data = self.cfg.map_data
        self.len = self._get_data_length(self.cfg.metadata_path)
        if num_samples:
            if num_samples > 0:
                self.len = min(num_samples, self.len)
        self.start = 0
        self.end = self.start + self.len
        self._cache = None
        self.regex = re.compile(r"""\,(?P<smiles>.+)""") # TODO make column selectable in regex

    def __len__(self):
        return self.len
    
    def _get_data_length(self, metadata_path: Optional[str] = None):
        """Try to read metadata file for length, otherwise fall back on scanning rows"""
        length = 0
        if metadata_path:
            assert os.path.exists(metadata_path), FileNotFoundError(f"Could not find metadata file {metadata_path}")
            base_filepath = os.path.splitext(os.path.basename(self.filepath))[0]
            with open(metadata_path, 'r') as fh:
                for line in fh:
                    data = line.strip().split(',')
                    if data[0] == base_filepath:
                        length = int(data[1])
                        break
        
        if length == 0:
            logging.info('Unable to determine dataset size from metadata. Falling back to countining lines.')
            with open(self.filepath, 'rb') as fh:
                for row, line in enumerate(fh):
                    pass
            length = row

        logging.info(f'Dataset {self.filepath} contains {length} molecules.')
        return length
    
    def _initialize_file(self, start):

        if self.map_data:
            self.fh = open(self.filepath, 'rb')
            self.fh.seek(0)
            fh_map = mmap.mmap(self.fh.fileno(), 0, prot=mmap.PROT_READ)
            fh_iter = iter(fh_map.readline, b'')
        else:
            fh_iter = iter(open(self.filepath, 'rb').readline, b'')
        _ = [next(fh_iter) for x in range(start + 1)] # scan to start row 
        self.fh_iter = fh_iter
        
    def parse_data(self, lines):
        if isinstance(lines, list):
            lines = b''.join(lines)
        lines = re.findall(self.regex, lines.decode('utf-8'))
        return lines
        
    def __exit__(self):
        if self.map_data:
            self.fh.close()


class MoleculeCsvDataset(Dataset, MoleculeCsvABCDataset):
    """Dataset that reads GPU-specific portion of data into memory from CSV file"""
    def __init__(self, filepath, cfg, trainer, num_samples=None):
        super().__init__(filepath=filepath, cfg=cfg, trainer=trainer, num_samples=num_samples)
        self._initialize_file(self.start)
        self._make_data_cache()
        
    def _make_data_cache(self):
        lines = [next(self.fh_iter) for x in range(self.len)]
        lines = self.parse_data(lines)
        assert len(lines) == self.len
        self._cache = lines
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self._cache[idx]


class MoleculeCsvIterableDataset(IterableDataset, MoleculeCsvABCDataset):
    def __init__(self, filepath, cfg, trainer, num_samples=None):
        super().__init__(filepath=filepath, cfg=cfg, trainer=trainer, num_samples=num_samples)
        
    def __iter__(self):  
        # Divide up for workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            iter_start = self.start + (worker_info.id * per_worker)
            iter_end = min(iter_start + per_worker, self.end)
        else:
            per_worker = self.len
            iter_start = self.start
            iter_end = self.end

        iter_len = iter_end - iter_start # handle smaller last batch
        self._initialize_file(iter_start)

        for _ in range(iter_len):
            mol = next(self.fh_iter)
            mol = self.parse_data(mol)[0]
            yield mol
