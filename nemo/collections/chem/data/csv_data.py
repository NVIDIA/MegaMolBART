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
    zinc: bool = False
    data_parallel_size: int = 1
    world_size: Optional[int] = 1
    num_samples: Optional[int] = None

# TODO remove
# @dataclass
# class MoleculeCsvCombinedDatasetConfig(DatasetConfig):
#     filepath: str = 'data.csv'
#     metadata_path: Optional[str] = None
#     zinc: bool = False
#     num_samples: Optional[int] = None

# @dataclass
# class MoleculeCsvStreamingDatasetConfig(DatasetConfig):
#     filepath: str = 'data.csv'
#     metadata_path: Optional[str] = None
#     zinc: bool = False
#     num_samples: Optional[int] = None

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
    def __init__(self, filepath: str, molecule_column_name: str, num_samples: int = None, metadata_path: str = None, world_size: int = 1, **kwargs):
        """
        Args:
            filepath (str): path to dataset file with compounds contained as smiles
        """
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
        self.world_size = world_size
        self.global_rank = None
        self.len = self.get_datasize(metadata_path) // world_size
        if num_samples:
            if num_samples > 0:
                self.len = min(num_samples, self.len)

    def __len__(self):
        return self.len

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
                # self.end_idx = self.begin_idx + self.len
            else:
                self.node_rank = 0
                self.local_rank = 0
                self.global_rank = 0
                self.begin_idx = 0

        # TODO remove
        # node_rank, local_rank, world_size = env_cp['NODE_RANK'], env_cp['LOCAL_RANK'], env_cp['WORLD_SIZE']
        # is_in_ddp_subprocess = env_cp['PL_IN_DDP_SUBPROCESS']
        # pl_trainer_gpus = env_cp['PL_TRAINER_GPUS']
        logging.info(f'LOADING DATA world_size {self.world_size} local_rank {self.local_rank} node_rank {self.node_rank} begin index {self.begin_idx}')
        
        line = linecache.getline(self.filepath, idx + self.begin_idx + 2)
        mol = line.strip().split(',')[self.index]
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

    def get_datasize(self, metadata_path: Optional[str] = None):
        # Get datasize
        length = 0

        if metadata_path:
            if not os.path.exists(metadata_path):
                assert FileNotFoundError(f"Could not find metadata file {metadata_path}")
            else:
                base_filepath = os.path.splitext(os.path.basename(self.filepath))
                with open(metadata_path, 'r') as fh:
                    data = line.strip().split(',')
                    if data[0] == base_filepath:
                        length = int(data[1])
        
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
        df = pd.read_csv(filepath)
        col = 'smiles' if zinc else 'canonical_smiles'
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
