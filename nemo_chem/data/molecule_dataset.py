# coding=utf-8

import os
import re
import math
import mmap
from typing import Optional
from dataclasses import dataclass

import torch
from nemo.core import Dataset, IterableDataset
from nemo.core.classes.dataset import DatasetConfig
from nemo.utils import logging
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import make_dataset

import time
__all__ = ['MoleculeCsvDatasetConfig', 'MoleculeDataset']


@dataclass
class MoleculeCsvDatasetConfig(DatasetConfig):
    filepath: str = 'data.csv'
    micro_batch_size: int = 1
    use_iterable: bool = False
    map_data: bool = False
    encoder_augment: bool = True
    encoder_mask: bool = False
    decoder_augment: bool = False
    canonicalize_input: bool = False
    metadata_path: Optional[str] = None
    num_samples: Optional[int] = None
    drop_last: bool = False
    shuffle: bool = False
    num_workers: Optional[int] = None
    pin_memory: bool = True # TODO: remove this if value is fixed


class MoleculeABCDataset():
    """Molecule base dataset that reads tokenized data from binarized input files."""
    
    def __init__(self, filepath: str, metadata_path: str = None, num_samples: int = None, map_data: bool = False): 
        """
        Args:
            filepath (str): path to dataset file with unmasked tokenized smiles
        """
        self.filepath = filepath
        self._cache = None

    def __len__(self):
        return self.len
    
    def _initialize_file(self):
        start_time = time.time()
        self.indexed_dataset = make_dataset(self.filepath,"mmap", skip_warmup=False)
        self.len = self.indexed_dataset.sizes.shape[0]
        assert self.indexed_dataset.sizes.shape[0] == self.indexed_dataset.doc_idx[-1]
        logging.info(' > finished creating indexed dataset in {:4f} ' 'seconds'.format(time.time() - start_time))

        logging.info(' > indexed dataset stats:')
        logging.info('    number of documents: {}'.format(self.indexed_dataset.doc_idx.shape[0] - 1))
        logging.info('    number of sentences: {}'.format(self.indexed_dataset.sizes.shape[0]))

    def __exit__(self):
        if self.map_data:
            self.fh.close()


class MoleculeDataset(Dataset, MoleculeABCDataset):
    """Dataset that reads GPU-specific portion of data into memory from CSV file"""
    def __init__(self, filepath: str, metadata_path: str = None, num_samples: int = None, map_data: bool = False, **kwargs):
        super().__init__(filepath=filepath, metadata_path=metadata_path, num_samples=num_samples, map_data=map_data)
        self._initialize_file()
        
    def __getitem__(self, idx):
        st = time.time()
        if torch.is_tensor(idx):
            idx = idx.item()
        return self.indexed_dataset.get(idx)
