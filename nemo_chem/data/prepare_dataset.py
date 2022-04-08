import torch
from nemo.utils import logging
from rdkit import Chem
from pysmilesutils.augment import SMILESAugmenter
from typing import List
import numpy as np

from nemo_chem.tokenizer import MolEncTokenizer
import time

__all__ = ['PrepareDataset']


class PrepareDataset:
    def __init__(self, tokenizer: MolEncTokenizer, max_seq_len: int, **kwargs):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len


    def _check_seq_len(self, tokens, mask):
        """ Warn user and shorten sequence if the tokens are too long, otherwise return original

        Args:
            tokens (List[List[str]]): List of token sequences
            mask (List[List[int]]): List of mask sequences

        Returns:
            tokens (List[List[str]]): List of token sequences (shortened, if necessary)
            mask (List[List[int]]): List of mask sequences (shortened, if necessary)
        """

        seq_len = max([len(ts) for ts in tokens])
        if seq_len > self.max_seq_len:
            tokens_short = [ts[:self.max_seq_len] for ts in tokens]
            mask_short = [ms[:self.max_seq_len] for ms in mask]
            return (tokens_short, mask_short)
        return (tokens, mask)

    def _canonicalize_smile(self, smile):
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile

    def convert_tokens_to_smiles(self, tokens, canonical: True):
        """Take in a token array and convert it back to a canonicalized smile"""
        smiles = self.tokenizer.detokenize(tokens)

        if canonical:
            canon_smiles = [self._canonicalize_smile(smile) for smile in smiles]
            return canon_smiles
        return smiles

    def _prepare_tokens(self, token_ids, mask_data: bool = False, canonicalize: bool = False):
        """Prepare tokens for encoder or decoder from batch of input SMILES strings

        Args:
            batch (List[str]): Batch of input SMILES strings
            tokenizer: Tokenizer instantiation.
            mask (bool, optional): Mask decoder tokens. Defaults to False.
            canonicalize (bool, optional): Canonicalize input SMILES. Defaults to False.
            smiles_augmenter (optional): Function to augment SMILES. Defaults to None.

        Returns:
            dict: token output
        """
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        #canonicalize all ids
        canon_target = self.convert_tokens_to_smiles(tokens, canonical=True) if canonicalize else []
        # padd and optionally mask the tokens
        tokens, masks = self.tokenizer.mask_tokens(tokens, empty_mask=mask_data)
        tokens, masks = self.tokenizer.pad_seqs(tokens, self.tokenizer.pad_token)
        tokens, masks = self._check_seq_len(tokens, masks)
        token_output = {
            "tokens": tokens,
            "pad_mask": masks,
            "target_smiles": canon_target
        }

        return token_output

    def collate_fn(self, batch):
        encoder_tokens = self._prepare_tokens(batch, mask_data=True, canonicalize=True)
        decoder_tokens = self._prepare_tokens(batch, mask_data=False, canonicalize=False)

        enc_token_ids = self.tokenizer.convert_tokens_to_ids(encoder_tokens['tokens'])
        enc_token_ids = torch.tensor(enc_token_ids).transpose(0, 1) # TODO why is this transpose done?
        enc_pad_mask = torch.tensor(encoder_tokens['pad_mask'], dtype=torch.bool).transpose(0, 1)

        dec_token_ids = self.tokenizer.convert_tokens_to_ids(decoder_tokens['tokens'])
        dec_token_ids = torch.tensor(dec_token_ids).transpose(0, 1)
        dec_pad_mask = torch.tensor(decoder_tokens['pad_mask'], dtype=torch.bool).transpose(0, 1)

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
