# coding=utf-8

from pathlib import Path
import os
import re
import mmap
import numpy as np
import pandas as pd
import csv
from typing import Optional
import linecache

import torch
# from torch.utils.data import Dataset
from nemo.core import Dataset
from nemo.core.classes.dataset import DatasetConfig
from nemo.utils import logging
from rdkit import Chem

from dataclasses import dataclass
from pysmilesutils.augment import SMILESAugmenter


@dataclass
class MoleculeCsvDatasetConfig(DatasetConfig):
    filepath: str = 'data.csv'
    metadata_path: Optional[str] = None
    num_samples: Optional[int] = None
    world_size: Optional[int] = 1
    num_gpus: Optional[int] = 1
    num_workers: Optional[int] = 0


class MoleculeABCDataset(Dataset):
    """Molecule base dataset that reads SMILES from the second column from CSV files."""
    def __init__(self, filepath: str, metadata_path: str = None, num_samples: int = None, 
                 world_size: int = 1, num_gpus: int = 1):
        """
        Args:
            filepath (str): path to dataset file with compounds contained as smiles
        """
        self.num_gpus = num_gpus # TODO find better way to get this data
        self.global_rank = None

        assert os.path.exists(filepath), FileNotFoundError(f"Could not find CSV file {filepath}")
        self.filepath = filepath             

        # Set length of dataset based on GPUs
        self.full_len = self._get_data_length(metadata_path) 
        self.len = self.full_len // world_size
        assert self.len * world_size <= self.full_len
        self.start = 0
        self.end = self.len
        self._cache = None
        
        if num_samples:
            if num_samples > 0:
                self.len = min(num_samples, self.len)
                
        self.aug = SMILESAugmenter() # TODO separate out augmenter
        self.regex = re.compile(r"""\,(?P<smiles>.+)""") # read second column

    def __len__(self):
        return self.len
    
    def _get_data_length(self, metadata_path: Optional[str] = None):
        """Try to read metadata file for length, otherwise fall back on scanning rows"""
        length = 0
        if metadata_path:
            assert os.path.exists(metadata_path), FileNotFoundError(f"Could not find metadata file {metadata_path}")
            base_filepath = os.path.splitext(os.path.basename(self.filepath))[0]
            with open(metadata_path, 'r') as fh: # TODO convert to read as binary?
                for line in fh:
                    data = line.strip().split(',')
                    if data[0] == base_filepath:
                        length = int(data[1])
                        break
        
        if length == 0:
            logging.info('Unable to determine dataset size from metadata. Falling back to line counting.')
            with open(self.filepath, 'r') as fh: # TODO convert to read as binary?
                for row, line in enumerate(fh):
                    pass
            length = row
        return length
    
    def _initialize_file(self):
        fh = open(self.filepath, 'rb')
        fh.seek(0)
        fh_map = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
        fh_iter = iter(fh_map.readline, "")
        _ = [next(fh_iter) for x in range(self.start + 1)] # scan to start row
        self.fh, self.fh_iter = fh, fh_iter
        
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
        self.fh.close()


class MoleculeDataset(MoleculeABCDataset):
    def __init__(self, filepath: str, metadata_path: str = None, num_samples: int = None, 
                 world_size: int = 1, num_gpus: int = 1, **kwargs):
        super().__init__(filepath=filepath, metadata_path=metadata_path, num_samples=num_samples, 
                     world_size=world_size, num_gpus=num_gpus)
        
    def _make_data_cache(self):
        lines = [next(self.fh_iter) for x in range(self.len)]
        lines = self.decoder(lines)
        from IPython import embed
        embed()
        assert len(lines) == self.len
        self._cache = lines
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self._cache is None:
            env = os.environ.copy()
            node_rank = int(env.get('NODE_RANK', 0))
            local_rank = int(env.get('LOCAL_RANK', 0))
            self.global_rank = (node_rank * self.num_gpus) + local_rank
            self.start = self.full_len * self.global_rank
            self.end = self.start + self.len
            self._initialize_file()
            self._make_data_cache()
                
        mol = self._cache[idx]
        enc_smi = self.augmenter(mol)
        dec_smi = self.augmenter(mol)
        output = {'encoder_smiles': enc_smi, 'decoder_smiles': dec_smi}
        return output


# TODO should class iterable dataset and make compatible with workers
class MoleculeIterableDataset(MoleculeABCDataset):
    def __init__(self, filepath: str, metadata_path: str = None, num_samples: int = None, 
                 world_size: int = 1, num_gpus: int = 1, **kwargs):
        super().__init__(filepath=filepath, metadata_path=metadata_path, num_samples=num_samples, 
                         world_size=world_size, num_gpus=num_gpus)
        # self.position = 0
        
    def __iter__(self):
        if self.global_rank is None:
            env = os.environ.copy()
            node_rank = int(env.get('NODE_RANK', 0))
            local_rank = int(env.get('LOCAL_RANK', 0))
            self.global_rank = (node_rank * self.num_gpus) + local_rank
            self.start = self.full_len * self.global_rank
            self._initialize_file()
  
        iter_start, iter_end = self.start, self.end
        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is None:  # single-process data loading, return the full iterator
        #     iter_start = self.start
        #     iter_end = self.end
        # else:  # in a worker process
        #     # split workload
        #     assert self.end - self.start >= self.len
        #     per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        #     worker_id = worker_info.id
        #     iter_start = self.start + worker_id * per_worker
        #     iter_end = min(iter_start + per_worker, self.end)
            
        while True:
            for _ in range(self.len):
                mol = next(self.fh_iter)
                mol = self.decoder(mol)[0]
                enc_smi = self.augmenter(mol)
                dec_smi = self.augmenter(mol)
                output = {'encoder_smiles': enc_smi, 'decoder_smiles': dec_smi}
                yield output


# class MoleculeCsvDataset(Dataset):
#     """Molecule dataset that reads from pre-split CSV files."""
#     def __init__(self, filepath: str, molecule_column_name: str, num_samples: int = None, **kwargs):
#         """
#         Args:
#             filepath (str): path to dataset file with compounds contained as smiles
#         """
#         self.aug = SMILESAugmenter()

#         if not os.path.exists(filepath):
#             assert FileNotFoundError(f"Could not find CSV file {filepath}")

#         index = 0
#         mols = []
#         with open(filepath) as csvfile:
#             csvreader = csv.reader(csvfile, delimiter=',')
#             for row,data in enumerate(csvreader):
#                 if row == 0:
#                     assert molecule_column_name in data
#                     index = data.index(molecule_column_name)
#                 else:
#                     mols.append(data[index])
        
#         if num_samples:
#             if num_samples > 0:
#                 num_samples = min(num_samples, len(mols))
#                 mols = mols[:num_samples]
#         self.mols = mols

#     def __len__(self):
#         return len(self.mols)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         mol = self.mols[idx]
#         try:
#             enc_smi = self.aug(mol)
#         except:
#             enc_smi = mol
#         try:
#             dec_smi = self.aug(mol)
#         except:
#             dec_smi = mol

#         output = {'encoder_smiles': enc_smi, 'decoder_smiles': dec_smi}
#         return output


# class MoleculeCsvStreamingDataset(Dataset):
#     """Molecule dataset that streams from pre-split CSV files."""
#     def __init__(self, filepath: str, molecule_column_name: str, metadata_path: str = None, num_samples: int = None, world_size: int = 1, num_gpus: int = 1, cache_data: bool = False, **kwargs):
#         """
#         Args:
#             filepath (str): path to dataset file with compounds contained as smiles
#         """
#         # TODO remove excess logging
#         self.world_size = world_size
#         self.num_gpus = num_gpus
#         self.global_rank = None
#         self.cache_data = cache_data
#         self.cache = None
#         self.aug = SMILESAugmenter()

#         if not os.path.exists(filepath):
#             assert FileNotFoundError(f"Could not find CSV file {filepath}")
#         else:
#             self.filepath = filepath

#         # Figure out column index
#         self.index = 0
#         with open(filepath, 'r') as fh:
#             line = next(fh)
#         data = line.strip().split(',')
#         assert molecule_column_name in data
#         self.index = data.index(molecule_column_name)

#         # Set length of dataset
#         self.full_len = self.get_data_length(metadata_path) 
#         self.len = self.full_len // self.world_size
#         assert self.len * self.world_size <= self.full_len
#         # logging.info(f'DATASET INIT world_size {self.world_size} full_len {self.full_len} len {self.len} total {self.len*self.world_size}') # TODO remove
#         if num_samples:
#             if num_samples > 0:
#                 self.len = min(num_samples, self.len)

#     def __len__(self):
#         return self.len

#     def _get_mol_from_file(self, idx):
#         row = idx + self.begin_idx
#         line = linecache.getline(self.filepath, row + 1) # linecache is one-based indexing # TODO switch to SEEK/SCAN
#         logging.info(f'DATASET ITEM idx {idx} begin_idx {self.begin_idx} row {row} len {self.len}') # TODO remove
#         logging.info(f'DATASET ITEM column {self.index} line {line}') # TODO remove
#         mol = line.strip().split(',')[self.index]
#         logging.info(f'DATASET ITEM mol {mol}') # TODO remove
#         return mol

#     def _make_data_cache(self):
#         self.cache = [self._get_mol_from_file(x) for x in range(self.len)]

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         if self.global_rank is None:
#             env = os.environ.copy()
#             self.node_rank = int(env.get('NODE_RANK', 0))
#             self.local_rank = int(env.get('LOCAL_RANK', 0))
#             self.global_rank = (self.node_rank * self.num_gpus) + self.local_rank
#             self.begin_idx = self.len * self.global_rank

#             # ALL CORRECT  # TODO remove
#             # logging.info(f'DATASET ITEM world_size {self.world_size} num_gpus {self.num_gpus} node_rank {self.node_rank} local_rank {self.local_rank} global_rank {self.global_rank}')
#             # logging.info(f'DATASET ITEM full_len {self.full_len} len {self.len} beg_idx {self.begin_idx} end_idx {self.begin_idx+self.len}')
        

#         if self.cache_data:
#             if not self.cache:
#                 self._make_data_cache()
#             mol = self.cache[idx]
#         else:
#             mol = self._get_mol_from_file(idx)

#         try:
#             enc_smi = self.aug(mol)
#         except:
#             enc_smi = mol
#         try:
#             dec_smi = self.aug(mol)
#         except:
#             dec_smi = mol

#         output = {'encoder_smiles': enc_smi, 'decoder_smiles': dec_smi}
#         return output

#     def get_data_length(self, metadata_path: Optional[str] = None):
#         length = 0
#         if metadata_path:
#             assert os.path.exists(metadata_path), FileNotFoundError(f"Could not find metadata file {metadata_path}")
#             base_filepath = os.path.splitext(os.path.basename(self.filepath))[0]
#             with open(metadata_path, 'r') as fh:
#                 for line in fh:
#                     data = line.strip().split(',')
#                     if data[0] == base_filepath:
#                         length = int(data[1])
#                         break
        
#         if length == 0:
#             logging.info('Unable to determine dataset size from metadata. Falling back to line counting.')
#             with open(self.filepath, 'r') as fh:
#                 for row,line in enumerate(fh):
#                     pass
#             length = row
#         return length


# class MoleculeCsvCombinedDataset(Dataset):
#     """Molecule dataset that reads train/val/testfrom a single DataFrame."""

#     def __init__(self, filepath: str, split: str, zinc: bool = False, **kwargs):
#         """
#         Args:
#             filepath (str): path to dataset file with compounds contained as smiles
#         """
#         assert split in ['train', 'val', 'test']
#         self.aug = SMILESAugmenter()

#         # TODO remove pandas
#         col = 'smiles' if zinc else 'canonical_smiles'
#         df = pd.read_csv(filepath)
#         mask = df['set'] == split
#         self.mols = df.loc[mask, col].tolist()

#     def __len__(self):
#         return len(self.mols)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         mol = self.mols[idx]
#         try:
#             enc_smi = self.aug(mol)
#         except:
#             enc_smi = mol
#         try:
#             dec_smi = self.aug(mol)
#         except:
#             dec_smi = mol
#         output = {'encoder_smiles': enc_smi, 'decoder_smiles': dec_smi}
#         return output
