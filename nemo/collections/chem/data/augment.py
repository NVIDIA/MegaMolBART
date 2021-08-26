from typing import Optional, Union, List
from pysmilesutils.augment import MolAugmenter
from rdkit import Chem
from nemo.utils import logging
from dataclasses import dataclass
from functools import partial
from copy import deepcopy

__all__ = ['prepare_tokens']


def smiles_augmeter_func(smiles: str, canonicalize_input: bool = False, 
                    smiles_augmenter: Optional[MolAugmenter] = None):
    """Regularize SMILES by coverting to RDKit mol objects and back

    Args:
        smiles (str): Input SMILES from dataset
        canonicalize_input (bool, optional): Canonicalize by default. Defaults to False.
        smiles_augmenter: Function to augment/randomize SMILES. Defaults to None
    """
    mol = Chem.MolFromSmiles(smiles)
    canon_smiles = Chem.MolToSmiles(mol, canonical=True) if canonicalize_input else smiles

    if smiles_augmenter:
        aug_mol = smiles_augmenter(mol)

        # There is a very rare possibility that RDKit will not be able to generate 
        # the SMILES for the augmented mol. In this case we just use the canonical 
        # mol to generate the SMILES
        try:
            aug_smiles = Chem.MolToSmiles(aug_mol, canonical=False)
        except RuntimeError:
            logging.info(f'Could not generate smiles for {smiles} after augmenting. Forcing canonicalization')
            aug_smiles = canon_smiles if canonicalize_input else Chem.MolToSmiles(mol, canonical=True)
    else:
        aug_smiles = Chem.MolToSmiles(mol, canonical=False)

    return aug_smiles, canon_smiles


def prepare_tokens(batch: List[str], tokenizer, mask: bool = False, 
                    canonicalize_input: bool = False,
                    smiles_augmenter: Optional[MolAugmenter] = None):
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
    smiles_func = partial(smiles_augmeter_func, canonicalize_input=canonicalize_input, 
                            smiles_augmenter=smiles_augmenter)
    smiles_list = [smiles_func(smiles) for smiles in batch]
    smiles = [x[0] for x in smiles_list]
    canon_targets = [x[1] for x in smiles_list]

    # Tokenize with padding and optional masking
    token_output = tokenizer.tokenize(smiles, pad=True, mask=mask)
    key = 'masked' if mask else 'original'
    token_label, mask_label = f'{key}_tokens', f'{key}_pad_masks'
    tokens = token_output[token_label]
    mask = token_output[mask_label]

    # Verify sequence length
    tokens, mask = check_seq_len(tokens, mask)

    token_output = {
        "tokens": tokens,
        "pad_mask": mask,
        "target_smiles": canon_targets
    }

    return token_output
