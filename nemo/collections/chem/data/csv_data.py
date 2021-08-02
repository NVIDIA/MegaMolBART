# coding=utf-8

from pathlib import Path
import os
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
    molecule_column_name: str = 'smiles'
    metadata_path: Optional[str] = None
    data_parallel_size: int = 1
    num_samples: Optional[int] = None
    num_workers: int = 0
    cache_data: bool = False
    zinc: bool = False


class MoleculeCsvDataset(Dataset):
    """Molecule dataset that reads from pre-split CSV files."""
    def __init__(self, filepath: str, molecule_column_name: str, num_samples: int = None, **kwargs):
        """
        Args:
            filepath (str): path to dataset file with compounds contained as smiles
        """
        self.aug = SMILESAugmenter()

        if not os.path.exists(filepath):
            assert FileNotFoundError(f"Could not find CSV file {filepath}")

        index = 0
        mols = []
        with open(filepath) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row,data in enumerate(csvreader):
                if row == 0:
                    assert molecule_column_name in data
                    index = data.index(molecule_column_name)
                else:
                    mols.append(data[index])
        
        if num_samples:
            if num_samples > 0:
                num_samples = min(num_samples, len(mols))
                mols = mols[:num_samples]
        self.mols = mols

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mol = self.mols[idx]
        try:
            enc_smi = self.aug(mol)
        except:
            enc_smi = mol
        try:
            dec_smi = self.aug(mol)
        except:
            dec_smi = mol

        output = {'encoder_smiles': enc_smi, 'decoder_smiles': dec_smi}
        return output


class MoleculeCsvStreamingDataset(Dataset):
    """Molecule dataset that streams from pre-split CSV files."""
    def __init__(self, filepath: str, molecule_column_name: str, metadata_path: str = None, num_samples: int = None, world_size: int = 1, cache_data: bool = False, **kwargs):
        """
        Args:
            filepath (str): path to dataset file with compounds contained as smiles
        """
        # TODO remove excess logging
        self.world_size = world_size
        self.global_rank = None
        self.cache_data = cache_data
        self.cache = None
        self.aug = SMILESAugmenter()

        if not os.path.exists(filepath):
            assert FileNotFoundError(f"Could not find CSV file {filepath}")
        else:
            self.filepath = filepath

        # Figure out column index
        self.index = 0
        with open(filepath, 'r') as fh:
            line = next(fh)
        data = line.strip().split(',')
        assert molecule_column_name in data
        self.index = data.index(molecule_column_name)

        # Set length of dataset
        self.full_len = self.get_data_length(metadata_path) 
        self.len = self.full_len // self.world_size
        assert self.len * self.world_size <= self.full_len
        logging.info(f'DATASET world_size {self.world_size} full_len {self.full_len} len {self.len} total {self.len*self.world_size}')
        if num_samples:
            if num_samples > 0:
                self.len = min(num_samples, self.len)

    def __len__(self):
        return self.len

    def _get_mol_from_file(self, idx):
        row = idx + self.begin_idx + 2
        line = linecache.getline(self.filepath, row)
        mol = line.strip().split(',')[self.index]
        return mol

    def _make_data_cache(self):
        self.cache = [self._get_mol_from_file(x) for x in range(self.len)]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.global_rank is None:
            env = os.environ.copy()
            if env.get('PL_IN_DDP_SUBPROCESS', False):
                num_gpus = len(env['PL_TRAINER_GPUS'].split(','))
                self.node_rank = int(env['NODE_RANK'])
                self.local_rank = int(env['LOCAL_RANK'])
                self.global_rank = (self.node_rank * num_gpus) + self.local_rank
                self.begin_idx = self.len * self.global_rank
                logging.info(f'DATASET DDP is running')
            else:
                num_gpus = 1
                self.node_rank = 0
                self.local_rank = 0
                self.global_rank = 0
                self.begin_idx = 0
                logging.info(f'DATASET DDP is NOT running')

            logging.info(f'DATASET world_size {self.world_size} num_gpus {num_gpus} node_rank {self.node_rank} local_rank {self.local_rank} global_rank {self.global_rank}')
            logging.info(f'DATASET full_len {self.full_len} len {self.len} beg_idx {self.begin_idx} end_idx {self.begin_idx+self.len}')
        

        if self.cache_data:
            if not self.cache:
                self._make_data_cache()
            mol = self.cache[idx]
        else:
            mol = self._get_mol_from_file(idx)

        try:
            enc_smi = self.aug(mol)
        except:
            enc_smi = mol
        try:
            dec_smi = self.aug(mol)
        except:
            dec_smi = mol

        output = {'encoder_smiles': enc_smi, 'decoder_smiles': dec_smi}
        return output

    def get_data_length(self, metadata_path: Optional[str] = None):
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
            with open(self.filepath, 'r') as fh:
                for row,line in enumerate(fh):
                    pass
            length = row
        return length


class MoleculeCsvCombinedDataset(Dataset):
    """Molecule dataset that reads train/val/testfrom a single DataFrame."""

    def __init__(self, filepath: str, split: str, zinc: bool = False, **kwargs):
        """
        Args:
            filepath (str): path to dataset file with compounds contained as smiles
        """
        assert split in ['train', 'val', 'test']
        self.aug = SMILESAugmenter()

        # TODO remove pandas
        col = 'smiles' if zinc else 'canonical_smiles'
        df = pd.read_csv(filepath)
        mask = df['set'] == split
        self.mols = df.loc[mask, col].tolist()

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mol = self.mols[idx]
        try:
            enc_smi = self.aug(mol)
        except:
            enc_smi = mol
        try:
            dec_smi = self.aug(mol)
        except:
            dec_smi = mol
        output = {'encoder_smiles': enc_smi, 'decoder_smiles': dec_smi}
        return output
