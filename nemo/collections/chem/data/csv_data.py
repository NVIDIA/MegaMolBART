# coding=utf-8

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

__all__ = ['MoleculeCsvDatasetConfig', 'MoleculeDataset', 'MoleculeIterableDataset']


@dataclass
class MoleculeCsvDatasetConfig():
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
    pin_memory: bool = True


class MoleculeABCDataset(MegatronDataset):
    """Molecule base dataset that reads SMILES from the second column from CSV files."""
    def __init__(self, filepath, cfg, trainer):
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
        if self.cfg.num_samples:
            if self.cfg.num_samples > 0:
                self.len = min(self.cfg.num_samples, self.len)
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


class MoleculeDataset(Dataset, MoleculeABCDataset):
    """Dataset that reads GPU-specific portion of data into memory from CSV file"""
    def __init__(self, filepath, cfg, trainer):
        super().__init__(filepath=filepath, cfg=cfg, trainer=trainer)
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


class MoleculeIterableDataset(IterableDataset, MoleculeABCDataset):
    def __init__(self, filepath, cfg, trainer):
        super().__init__(filepath=filepath, cfg=cfg, trainer=trainer)
        
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
