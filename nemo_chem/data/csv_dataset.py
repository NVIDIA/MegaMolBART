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
from .data_index import build_index_files

try:
    from apex.transformer.parallel_state import get_rank_info
    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

__all__ = ['MoleculeCsvDatasetConfig', 'MoleculeCsvDataset']

@dataclass
class MoleculeCsvDatasetConfig():
    dataset_path: str = ''
    dataset_files: str = 'data.csv'
    dataset_type: str = 'zinc_csv'
    newline_int: int = 10
    header_lines: int = 1
    skip_lines: int = 0
    data_col: int = 1
    data_sep: str = ','
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


class MoleculeCsvDataset(Dataset):
    """
    Allow per-line lazy access to multiple text files using numpy memmap.
    """
    def __init__(self,
                 dataset_paths,
                 cfg,
                 workers=None):

        if len(dataset_paths) < 1:
            raise ValueError("Dataset file list must contain at least one file name")

        super().__init__()

        # TODO not all of these need their state set
        self._header_lines = cfg.get('header_lines') # skip first N lines
        self._skip_lines = cfg.get('skip_lines')
        self._data_col = cfg.get('data_col')
        self._data_sep = cfg.get('data_sep')
        self._newline_int = cfg.get('newline_int')
        self._workers = workers

        self.mdata_midx_size_list = None
        
        # load all files into memmap
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        # is_global_rank_0 = (not is_distributed) or (is_distributed and torch.distributed.get_rank() == 0)
        rank_indexes = get_rank_info() # will return 0,0,0 if no dp/mp
        is_global_rank_0 = sum([0 if r is None else r for r in rank_indexes]) == 0

        if is_global_rank_0:
            
            data_parallel_rank, tensor_parallel_rank, pipeline_parallel_rank, virtual_pipeline_parallel_rank = rank_indexes
            logging.info(f'Building memory mapped indexes on tensor_parallel_rank {tensor_parallel_rank}, pipeline_parallel_rank {pipeline_parallel_rank}, data_parallel_rank {data_parallel_rank}, virtual_pipeline_parallel_rank {virtual_pipeline_parallel_rank} ')
            start_time = time.time()
            build_index_files(dataset_paths, self._newline_int, workers=self._workers)
            logging.info(f'Time to build memory mapped indexes: {time.time() - start_time}')
        else:
            logging.info(f'Building memory mapped indexes was skipped on this process because it is not global rank 0')

        if is_distributed:
            torch.distributed.barrier()

        logging.info(f"Loading data files")
        mdata_midx_size_list = [self.load_file(fn) for fn in dataset_paths]

        logging.info("Computing global indices")
        joint_midx = [0]
        for i in range(len(mdata_midx_size_list)):
            midx = mdata_midx_size_list[i][1]
            joint_midx.append(joint_midx[-1] + (len(midx) - self._header_lines))

        self.joint_midx = joint_midx
        self.mdata_midx_size_list = mdata_midx_size_list

    def __del__(self):
        if self.mdata_midx_size_list:
            for mdata, midx, size in self.mdata_midx_size_list:
                mdata._mmap.close()

    def __len__(self):
        return self.joint_midx[-1]

    def __getitem__(self, idx):
        """
        Return a string
        """
        # Identify the file containing the record
        file_id = 0
        for end_idx in self.joint_midx[1:]:
            if idx < end_idx:
                break
            file_id += 1
        file_row = idx - self.joint_midx[file_id]

        rec_start = self.mdata_midx_size_list[file_id][1][file_row]
        rec_end = self.mdata_midx_size_list[file_id][1][file_row + 1 + self._skip_lines]
        data = self.mdata_midx_size_list[file_id][0][rec_start:rec_end].tobytes().decode("ascii")
        return data.split(self._data_sep)[self._data_col]

    @staticmethod
    def load_file(fn):
        """
        Loads a text file as np.int8.
        Returns:
            mdata - memorymap of np.int8
            midx - indices pointing to the end-of-line (or end of file) position
        """
        logging.info(f"Loading {fn}")
        idx_fn = fn + ".idx"

        # create data map
        mdata = np.memmap(fn, dtype=np.uint8, mode='r')

        if os.path.exists(idx_fn):
            idx_dict = pickle.load(open(idx_fn, 'rb'))
            midx = idx_dict['midx']
            size = idx_dict['size']
        else:
            raise ValueError(f'Memory map for {fn} is not found')

        return (mdata, midx, size)
