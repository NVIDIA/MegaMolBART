# coding=utf-8

from pathlib import Path
import os
import re
import math
import mmap
import numpy as np
import pandas as pd
import csv
from typing import Optional
import linecache

import torch
# from torch.utils.data import Dataset
from nemo.core import Dataset, IterableDataset
from nemo.core.classes.dataset import DatasetConfig
from nemo.utils.app_state import AppState
from nemo.utils import logging
from rdkit import Chem

from dataclasses import dataclass
from pysmilesutils.augment import SMILESAugmenter


@dataclass
class MoleculeCsvDatasetConfig(DatasetConfig):
    filepath: str = 'data.csv'
    metadata_path: Optional[str] = None
    num_samples: Optional[int] = None
    num_workers: Optional[int] = 0
    use_iterable: Optional[bool] = False
    map_data: Optional[bool] = False
    world_size: Optional[int] = 1, 
    global_rank: Optional[int] = 0


class MoleculeABCDataset():
    """Molecule base dataset that reads SMILES from the second column from CSV files."""
    
    def __init__(self, filepath: str, metadata_path: str = None, num_samples: int = None, map_data: bool = False,
                world_size: int = 1, global_rank: int = 0): 
        """
        Args:
            filepath (str): path to dataset file with compounds contained as smiles
        """
        assert os.path.exists(filepath), FileNotFoundError(f"Could not find CSV file {filepath}")
        self.filepath = filepath
        self.map_data = map_data

        # Get GPU global_rank
        # app_state = AppState() # TODO fix me
        # self.global_rank = app_state._global_rank
        # world_size = app_state._world_size
        self.global_rank = global_rank

        # Set length of dataset based on GPUs
        self.full_len = self._get_data_length(metadata_path) 
        self.len = self.full_len // world_size
        assert self.len * world_size <= self.full_len
        self.start = self.len * self.global_rank
        
        if num_samples:
            if num_samples > 0:
                self.len = min(num_samples, self.len)

        self.end = self.start + self.len
        self._cache = None
                
        self.aug = SMILESAugmenter() # TODO create augmenter class and add augmentation probability
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
            logging.info('Unable to determine dataset size from metadata. Falling back to line counting.')
            with open(self.filepath, 'rb') as fh:
                for row, line in enumerate(fh):
                    pass
            length = row

        logging.info(f'Dataset {self.filepath} calculated to be {length} lines.')
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
        
    def decoder(self, lines):
        if isinstance(lines, list):
            lines = b''.join(lines)
        lines = re.findall(self.regex, lines.decode('utf-8'))
        return lines
    
    def augmenter(self, mol):
        try:
            aug_smi = self.aug(mol)
        except:
            aug_smi = mol
        return aug_smi
        
    def __exit__(self):
        if self.map_data:
            self.fh.close()


class MoleculeDataset(Dataset, MoleculeABCDataset):
    """Dataset that reads GPU-specific portion of data into memory from CSV file"""
    def __init__(self, filepath: str, metadata_path: str = None, num_samples: int = None, map_data: bool = False, **kwargs):
        super().__init__(filepath=filepath, metadata_path=metadata_path, num_samples=num_samples, map_data=map_data)
        self._initialize_file(self.start)
        self._make_data_cache()
        logging.info(f'DATASET {filepath}, {self.global_rank}, {self.start}, {self.end}')  # TODO REMOVE
        
    def _make_data_cache(self):
        lines = [next(self.fh_iter) for x in range(self.len)]
        lines = self.decoder(lines)
        assert len(lines) == self.len
        self._cache = lines
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mol = self._cache[idx]            
        enc_smi = self.augmenter(mol)
        dec_smi = self.augmenter(mol)
        output = {'encoder_smiles': enc_smi, 'decoder_smiles': dec_smi}
        return output


class MoleculeIterableDataset(IterableDataset, MoleculeABCDataset):
    def __init__(self, filepath: str, metadata_path: str = None, num_samples: int = None, **kwargs):
        super().__init__(filepath=filepath, metadata_path=metadata_path, num_samples=num_samples, map_data=False)
        logging.info(f'DATASET {filepath}, {self.global_rank}, {self.start}, {self.end}') # TODO REMOVE
        
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
            mol = self.decoder(mol)[0]
            enc_smi = self.augmenter(mol)
            dec_smi = self.augmenter(mol)
            output = {'encoder_smiles': enc_smi, 'decoder_smiles': dec_smi}
            yield output
