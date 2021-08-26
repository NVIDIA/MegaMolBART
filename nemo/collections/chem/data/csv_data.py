# coding=utf-8

import os
import re
import math
import mmap
from typing import Optional, List
from dataclasses import dataclass
from pysmilesutils.augment import MolAugmenter

import torch
# from torch.utils.data import Dataset
from nemo.core import Dataset, IterableDataset
from nemo.core.classes.dataset import DatasetConfig
from nemo.utils import logging
from .augment import prepare_tokens

__all__ = ['MoleculeCsvDatasetConfig', 'MoleculeDataset', 'MoleculeIterableDataset', 'collate_molecule_dataset']


def collate_molecule_dataset(batch, tokenizer, 
                             decoder_augment: bool, encoder_augment: bool = True, 
                             encoder_mask: bool = False, canonicalize_target: bool = False,
                             smiles_augmenter: Optional[MolAugmenter] = None):
    
    encoder_augmenter = smiles_augmenter if encoder_augment else None
    encoder_tokens = prepare_tokens(batch, tokenizer, mask=encoder_mask,
                                          canonicalize_input=canonicalize_target,
                                          smiles_augmenter=encoder_augmenter)

    decoder_augmenter = smiles_augmenter if decoder_augment else None
    decoder_tokens = prepare_tokens(batch, tokenizer, mask=False,
                                          canonicalize_input=False,
                                          smiles_augmenter=decoder_augmenter)

    enc_token_ids = tokenizer.convert_tokens_to_ids(enc_tokens['tokens'])
    enc_token_ids = torch.tensor(enc_token_ids).transpose(0, 1) # TODO why is this transpose done?
    enc_pad_mask = torch.tensor(enc_tokens['pad_mask'], dtype=torch.bool).transpose(0, 1)

    dec_token_ids = tokenizer.convert_tokens_to_ids(dec_tokens['tokens'])
    dec_token_ids = torch.tensor(dec_token_ids).transpose(0, 1)
    dec_pad_mask = torch.tensor(dec_tokens['pad_mask'], dtype=torch.bool).transpose(0, 1)

    collate_output = {
        "encoder_input": enc_token_ids,
        "encoder_pad_mask": enc_pad_mask,
        "decoder_input": dec_token_ids[:-1, :],
        "decoder_pad_mask": dec_pad_mask[:-1, :],
        "target": dec_token_ids.clone()[1:, :],
        "target_pad_mask": dec_pad_mask.clone()[1:, :],
        "target_smiles": encoder_tokens['target_smiles']
    }

    return collate_output


@dataclass
class MoleculeCsvDatasetConfig(DatasetConfig):
    filepath: str = 'data.csv'
    batch_size: int = 1
    use_iterable: bool = False
    map_data: bool = False
    encoder_augment: bool = True
    encoder_mask: bool = False
    decoder_augment: bool = False
    canonicalize_target_smiles: bool = False
    metadata_path: Optional[str] = None
    num_samples: Optional[int] = None


class MoleculeABCDataset():
    """Molecule base dataset that reads SMILES from the second column from CSV files."""
    
    def __init__(self, filepath: str, metadata_path: str = None, num_samples: int = None, map_data: bool = False): 
        """
        Args:
            filepath (str): path to dataset file with compounds contained as smiles
        """
        assert os.path.exists(filepath), FileNotFoundError(f"Could not find CSV file {filepath}")
        self.filepath = filepath
        self.map_data = map_data
        self.len = self._get_data_length(metadata_path)
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
           
    def __exit__(self):
        if self.map_data:
            self.fh.close()


class MoleculeDataset(Dataset, MoleculeABCDataset):
    """Dataset that reads GPU-specific portion of data into memory from CSV file"""
    def __init__(self, filepath: str, metadata_path: str = None, num_samples: int = None, map_data: bool = False, **kwargs):
        super().__init__(filepath=filepath, metadata_path=metadata_path, num_samples=num_samples, map_data=map_data)
        self._initialize_file(self.start)
        self._make_data_cache()
        
    def _make_data_cache(self):
        lines = [next(self.fh_iter) for x in range(self.len)]
        lines = self.decoder(lines)
        assert len(lines) == self.len
        self._cache = lines
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self._cache[idx]


class MoleculeIterableDataset(IterableDataset, MoleculeABCDataset):
    def __init__(self, filepath: str, metadata_path: str = None, num_samples: int = None, **kwargs):
        super().__init__(filepath=filepath, metadata_path=metadata_path, num_samples=num_samples, map_data=False)
        
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
            yield mol


# DEPRECATED
# def collate_fn(batch, augmenter, tokenizer, max_seq_len):
#     """ Used by DataLoader to concatenate/collate inputs."""
#     encoder_smiles = copy(batch)
#     decoder_smiles = batch

#     encoder_smiles = [augmenter(x) for x in encoder_smiles]
#     decoder_smiles = [augmenter(x) for x in decoder_smiles]

#     enc_token_output = tokenizer.tokenize(encoder_smiles, mask=True, pad=True)
#     dec_token_output = tokenizer.tokenize(decoder_smiles, pad=True)

#     enc_mask = enc_token_output['masked_pad_masks']
#     enc_tokens = enc_token_output['masked_tokens']
#     dec_tokens = dec_token_output['original_tokens']
#     dec_mask = dec_token_output['original_pad_masks']

#     (enc_tokens, enc_mask) = check_seq_len(enc_tokens, enc_mask, max_seq_len)
#     (dec_tokens, dec_mask) = check_seq_len(dec_tokens, dec_mask, max_seq_len)

#     enc_token_ids = tokenizer.convert_tokens_to_ids(enc_tokens)
#     dec_token_ids = tokenizer.convert_tokens_to_ids(dec_tokens)

#     enc_token_ids = torch.tensor(enc_token_ids).transpose(0, 1)
#     enc_pad_mask = torch.tensor(enc_mask,
#                                 dtype=torch.int64).transpose(0, 1)
#     dec_token_ids = torch.tensor(dec_token_ids).transpose(0, 1)
#     dec_pad_mask = torch.tensor(dec_mask,
#                                 dtype=torch.int64).transpose(0, 1)

#     collate_output = {
#         'encoder_input': enc_token_ids,
#         'encoder_pad_mask': enc_pad_mask,
#         'decoder_input': dec_token_ids[:-1, :],
#         'decoder_pad_mask': dec_pad_mask[:-1, :],
#         'target': dec_token_ids.clone()[1:, :],
#         'target_pad_mask': dec_pad_mask.clone()[1:, :],
#         'target_smiles': decoder_smiles,
#         }

#     return collate_output