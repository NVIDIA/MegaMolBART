import torch
from nemo.utils import logging
from rdkit import Chem
from pysmilesutils.augment import SMILESAugmenter
from typing import List
import numpy as np

from nemo_chem.tokenizer import MolEncTokenizer


__all__ = ['MoleculeEnumeration']


class MoleculeEnumeration:
    def __init__(self, tokenizer: MolEncTokenizer, seq_length: int,
                encoder_augment: bool, encoder_mask: bool, 
                decoder_augment, canonicalize_input, **kwargs):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.encoder_augment = encoder_augment
        self.encoder_mask = encoder_mask
        self.decoder_augment = decoder_augment
        self.canonicalize_input = canonicalize_input
        # self.aug = CanonicalSMILESAugmenter().randomize_mol_restricted

    def _smiles_augmeter_func(self, smiles: str, augment_data: bool):
        """Regularize SMILES by coverting to RDKit mol objects and back

        Args:
            smiles (str): Input SMILES from dataset
            canonicalize_input (bool, optional): Canonicalize by default. Defaults to False.
            smiles_augmenter: Function to augment/randomize SMILES. Defaults to None
        """
        mol = Chem.MolFromSmiles(smiles)
        canon_smiles = Chem.MolToSmiles(mol, canonical=True) if self.canonicalize_input else smiles

        if augment_data:
            # aug_mol = self.aug(mol)
            atom_order: List[int] = list(range(mol.GetNumAtoms()))
            np.random.shuffle(atom_order)
            aug_mol = Chem.RenumberAtoms(mol, atom_order) # TODO how to use PySMILESutils for this

            # There is a very rare possibility that RDKit will not be able to generate 
            # the SMILES for the augmented mol. In this case we just use the canonical 
            # mol to generate the SMILES
            try:
                aug_smiles = Chem.MolToSmiles(aug_mol, canonical=False)
            except RuntimeError:
                logging.info(f'Could not generate smiles for {smiles} after augmenting. Forcing canonicalization')
                aug_smiles = canon_smiles if self.canonicalize_input else Chem.MolToSmiles(mol, canonical=True)
        else:
            aug_smiles = Chem.MolToSmiles(mol, canonical=False)

        return aug_smiles, canon_smiles

    def _check_seq_len(self, tokens: List[List[str]], mask: List[List[int]]):
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
            mask_short = [ms[:self.seq_length] for ms in mask]
            return (tokens_short, mask_short)
        return (tokens, mask)

    def _prepare_tokens(self, batch: List[str], augment_data: bool, mask_data: bool = False):
        """Prepare tokens for encoder or decoder from batch of input SMILES strings

        Args:
            batch (List[str]): Batch of input SMILES strings
            tokenizer: Tokenizer instantiation.
            mask (bool, optional): Mask decoder tokens. Defaults to False.
            canonicalize_input (bool, optional): Canonicalize input SMILES. Defaults to False.
            smiles_augmenter (optional): Function to augment SMILES. Defaults to None.

        Returns:
            dict: token output
        """
        # Perform augmentation
        smiles_list = [self._smiles_augmeter_func(smiles, augment_data=augment_data) for smiles in batch]
        smiles = [x[0] for x in smiles_list]
        canon_targets = [x[1] for x in smiles_list]

        # Tokenize with padding and optional masking
        token_output = self.tokenizer.tokenize(smiles, pad=True, mask=mask_data)
        key = 'masked' if mask_data else 'original'
        token_label, mask_label = f'{key}_tokens', f'{key}_pad_masks'
        tokens = token_output[token_label]
        mask = token_output[mask_label]

        # Verify sequence length
        tokens, mask = self._check_seq_len(tokens, mask)

        token_output = {
            "tokens": tokens,
            "pad_mask": mask,
            "target_smiles": canon_targets
        }

        return token_output

    def collate_fn(self, batch: List[str], label_pad: int = -1):
        encoder_tokens = self._prepare_tokens(batch, augment_data=self.encoder_augment, mask_data=self.encoder_mask)
        decoder_tokens = self._prepare_tokens(batch, augment_data=self.decoder_augment, mask_data=False)

        # Dimensions required by NeMo: [batch, sequence/padding] 
        enc_token_ids = self.tokenizer.convert_tokens_to_ids(encoder_tokens['tokens'])
        enc_token_ids = torch.tensor(enc_token_ids, dtype=torch.int64) # TODO ensure transpose is removed
        
        enc_pad_mask = torch.tensor(encoder_tokens['pad_mask'], dtype=torch.bool) # TODO ensure transpose is removed
        enc_pad_mask = ~enc_pad_mask # TODO ensure active = True, padded = False for NeMo, must invert in the tokenizer created
        enc_pad_mask = enc_pad_mask.type(torch.int64)

        dec_token_ids = self.tokenizer.convert_tokens_to_ids(decoder_tokens['tokens'])
        dec_token_ids = torch.tensor(dec_token_ids, dtype=torch.int64) # TODO ensure transpose is removed
        
        dec_pad_mask = torch.tensor(decoder_tokens['pad_mask'], dtype=torch.bool) # TODO ensure transpose is removed
        labels = dec_token_ids.clone()
        labels[dec_pad_mask] = label_pad # TODO note this won't work if mask sign isn't corrected

        dec_pad_mask = ~dec_pad_mask # TODO ensure active = True, padded = False for NeMo
        dec_pad_mask = dec_pad_mask.type(torch.int64)

        loss_mask = dec_pad_mask.clone()

        collate_output = {'text_enc': enc_token_ids,
                          'text_dec': dec_token_ids,
                          'labels': labels,
                          'loss_mask': loss_mask,
                          'enc_mask': enc_pad_mask,
                          'dec_mask': dec_pad_mask}

        return collate_output
