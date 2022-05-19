import torch
from nemo.utils import logging
from rdkit import Chem
from pysmilesutils.augment import SMILESAugmenter
from typing import List
import numpy as np
import math
from nemo_chem.tokenizer import MolEncTokenizer
import time

__all__ = ['PrepareDataset']


class PrepareDataset:
    def __init__(self, tokenizer: MolEncTokenizer, seq_length: int,
            pad_size_divisible_by_8: bool, **kwargs):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.pad_size_divisible_by_8 = pad_size_divisible_by_8

    def _check_seq_len(self, tokens):
        """ Warn user and shorten sequence if the tokens are too long, otherwise return original

        Args:
            tokens (List[List[str]]): List of token sequences
            mask (List[List[int]]): List of mask sequences

        Returns:
            tokens (List[List[str]]): List of token sequences (shortened, if necessary)
            mask (List[List[int]]): List of mask sequences (shortened, if necessary)
        """

        seq_len = max([len(ts) for ts in tokens])
        if seq_len > self.seq_length:
            tokens_short = [ts[:self.seq_length] for ts in tokens]
            return tokens_short
        return tokens

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

    def _pad_seqs(self, seqs, pad_token):
        pad_length = max([len(seq) for seq in seqs])
        if self.pad_size_divisible_by_8:
            pad_length = int(math.ceil(pad_length/8) * 8)
        padded = [np.append(seq, np.array([pad_token] * (pad_length - len(seq)))) for seq in seqs]
        masks = [([1] * len(seq)) + ([0] * (pad_length - len(seq))) for seq in seqs] # 1/True = Active, 0/False = Inactive
        return padded, masks

    def _prepare_tokens(self, token_ids, canonicalize: bool = False):
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
        canon_target = self.convert_tokens_to_smiles(tokens, canonical=False)
        # pad and optionally mask the tokens
        token_ids  = self._check_seq_len(token_ids)
        token_output = {
            "token_ids": token_ids,
            "target_smiles": canon_target
        }

        return token_output

    def collate_fn(self, batch: List[np.array], label_pad: int = -1):
        encoder_tokens = self._prepare_tokens(batch, canonicalize=False)
        enc_token_ids, enc_pad_mask = self._pad_seqs(encoder_tokens['token_ids'], self.tokenizer.pad_id)
        enc_token_ids = torch.tensor(enc_token_ids, dtype=torch.int64)  #converting a list into torch tensor is very slow, convert to np.array first
        enc_pad_mask = torch.tensor(enc_pad_mask, dtype=torch.int64)

        decoder_tokens = self._prepare_tokens(batch, canonicalize=False)
        label_ids = [sample + [self.tokenizer.eos_id] for sample in decoder_tokens['token_ids']] # assign label_ids before adding bos_id to decoder
        dec_token_ids = [[self.tokenizer.bos_id] + sample for sample in decoder_tokens['token_ids']]
        dec_token_ids, dec_pad_mask = self._pad_seqs(dec_token_ids, self.tokenizer.pad_id)
        dec_token_ids = torch.tensor(dec_token_ids, dtype=torch.int64)
        dec_pad_mask = torch.tensor(dec_pad_mask, dtype=torch.int64)

        label_token_ids, loss_mask = self._pad_seqs(label_ids, self.tokenizer.pad_id)
        label_token_ids = torch.tensor(label_token_ids, dtype=torch.int64)
        loss_mask = torch.tensor(loss_mask, dtype=torch.int64)
        label_token_ids[~loss_mask.to(torch.bool)] = label_pad

        collate_output = {
            "text_enc": enc_token_ids,
            "enc_mask": enc_pad_mask,
            "text_dec": dec_token_ids,
            "dec_mask": dec_pad_mask,
            'labels': label_token_ids,
            'loss_mask': loss_mask,
            'target_smiles': encoder_tokens['target_smiles']} # smiles strings
        return collate_output
