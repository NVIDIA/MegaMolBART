# coding=utf-8

from pathlib import Path
import os
import numpy as np
import pandas as pd
import csv
from typing import Optional

import torch
from torch.utils.data import Dataset
from nemo.core.classes.dataset import DatasetConfig
from rdkit import Chem

from dataclasses import dataclass
from pysmilesutils.augment import SMILESAugmenter


@dataclass
class MoleculeCsvDatasetConfig(DatasetConfig):
    filepath: str = 'data.csv'
    molecule_column_name: str = 'smiles'
    num_samples: Optional[int] = None


@dataclass
class MoleculeCsvCombinedDatasetConfig(DatasetConfig):
    filepath: str = 'data.csv'
    zinc: bool = False
    num_samples: Optional[int] = None

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


class MoleculeCsvCombinedDataset(Dataset):
    """Molecule dataset that reads train/val/testfrom a single DataFrame."""
    # TODO remove pandas

    def __init__(self, filepath: str, split: str, zinc: bool = False, **kwargs):
        """
        Args:
            filepath (str): path to dataset file with compounds contained as smiles
        """
        assert split in ['train', 'val', 'test']
        self.aug = SMILESAugmenter()

        # TODO: switch to line by line parsing
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
