# coding=utf-8

from pathlib import Path
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from rdkit import Chem

from pysmilesutils.augment import SMILESAugmenter

try:
    from megatron.data.samplers import DistributedBatchSampler
except:
    pass

from megatron import mpu, get_args

from nemo.collections.chem.parts.util import DEFAULT_CHEM_TOKEN_START, DEFAULT_VOCAB_PATH, DEFAULT_MAX_SEQ_LEN, REGEX
from nemo.collections.chem.tokenizer.tokenizer import load_tokenizer

# default_tokenizer = load_tokenizer(vocab_path=DEFAULT_VOCAB_PATH, chem_token_start=DEFAULT_CHEM_TOKEN_START, regex=REGEX)


# def check_seq_len(tokens, mask, max_seq_len=DEFAULT_MAX_SEQ_LEN):
#     """ Warn user and shorten sequence if the tokens are too long, otherwise return original

#     Args:
#         tokens (List[List[str]]): List of token sequences
#         mask (List[List[int]]): List of mask sequences

#     Returns:
#         tokens (List[List[str]]): List of token sequences (shortened, if necessary)
#         mask (List[List[int]]): List of mask sequences (shortened, if necessary)
#     """

#     seq_len = max([len(ts) for ts in tokens])
#     if seq_len > max_seq_len:
#         tokens_short = [ts[:max_seq_len] for ts in tokens]
#         mask_short = [ms[:max_seq_len] for ms in mask]
#         return (tokens_short, mask_short)
#     return (tokens, mask)


# def collate_fn(batch):
#     """ Used by DataLoader to concatenate/collate inputs."""

#     encoder_smiles = [x['encoder_smiles'][0] for x in batch]
#     decoder_smiles = [x['decoder_smiles'][0] for x in batch]
#     enc_token_output = default_tokenizer.tokenize(encoder_smiles, mask=True,
#             pad=True)
#     dec_token_output = default_tokenizer.tokenize(decoder_smiles, pad=True)

#     enc_mask = enc_token_output['masked_pad_masks']
#     enc_tokens = enc_token_output['masked_tokens']
#     dec_tokens = dec_token_output['original_tokens']
#     dec_mask = dec_token_output['original_pad_masks']

#     (enc_tokens, enc_mask) = check_seq_len(enc_tokens, enc_mask)
#     (dec_tokens, dec_mask) = check_seq_len(dec_tokens, dec_mask)

#     enc_token_ids = default_tokenizer.convert_tokens_to_ids(enc_tokens)
#     dec_token_ids = default_tokenizer.convert_tokens_to_ids(dec_tokens)
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


class MoleculeDataset(Dataset):
    """Simple Molecule dataset that reads from a single DataFrame."""

    def __init__(self, filepath: str, split: str, zinc: bool = False):
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

# Not for use with NeMo
class MoleculeDataLoader(object):

    """Loads data from a csv file containing molecules."""

    def __init__(
        self,
        file_path,
        batch_size=32,
        num_buckets=20,
        num_workers=32,
        vocab_path=DEFAULT_VOCAB_PATH, 
        chem_token_start=DEFAULT_CHEM_TOKEN_START, 
        regex=REGEX
        ):

        path = Path(file_path)
        if path.is_dir():
            self.df = self._read_dir_df(file_path)
        else:
            self.df = pd.read_csv(path)

        train_dataset = MoleculeDataset(self.df, split='train', zinc=True)
        val_dataset = MoleculeDataset(self.df, split='val', zinc=True)
        self.tokenizer = load_tokenizer(vocab_path, chem_token_start, regex)

        world_size = \
            torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
        rank = \
            torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        sampler = torch.utils.data.SequentialSampler(train_dataset)
        batch_sampler = DistributedBatchSampler(sampler, batch_size,
                True, rank, world_size)

        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                batch_sampler=batch_sampler, num_workers=num_workers,
                pin_memory=True, collate_fn=collate_fn)
        self.val_loader = torch.utils.data.DataLoader(val_dataset,
                num_workers=num_workers, pin_memory=True,
                collate_fn=collate_fn)

    def get_data(self):
        return (self.train_loader, self.val_loader)

    def _read_dir_df(self, path):
        args = get_args()
        names = os.listdir(path)
        m = len(names)
        world_size = max(mpu.get_data_parallel_world_size(), args.world_size)
        rank = max(mpu.get_data_parallel_rank(), args.rank)
        partition = int(m/world_size) + 1
        partition = max(partition, 10)
        idx = partition*rank % m
        selected_names = names[idx:(idx+10)]
        dfs = [pd.read_csv(path + '/' + f) for f in selected_names]

        zinc_df = pd.concat(dfs, ignore_index=True, copy=False)
        return zinc_df
