
"Preprocess megamolbart data"
"Take text files as input and output tokenized data in binarized format"
##TODO Merge it with the CSV format data preprocessing script

import argparse
import gzip
import json
import os
import sys
import time
from glob import glob
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from nemo.core.config.modelPT import NemoConfig
import numpy as np
#import ftfy
import torch
from nemo_chem.tokenizer import MolEncTokenizer, MolEncTokenizerFromVocabFileConfig
from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset
from rdkit import Chem
from pysmilesutils.augment import SMILESAugmenter
from functools import partial
import multiprocessing

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        dest="input",
                        help="Path to a csv file or Path to a folder where input files \
                        exist. We recursively go through and \
                        consolidate all CSV files into one \
                        binarized output file. We expect the first row\
                        to be a header with zinc_id and smiles column names.\
                        We only parse the smiles column.")
    parser.add_argument("--config", type=str, required=True,
                        dest="config",
                        help="The training config file path. Tokenizer params will be\
                              extracted and used for preparing the data.")
    parser.add_argument("--out-dir", type=str, required=False,
                        dest="out_dir",
                        help="Path where the output is to be stored. This should not be a file name,\
                        just a directory where the output is to be saved.",
                        default=".")
    parser.add_argument("--batch-size", type=int, required=False,
                        dest="batch_size",
                        help="Number of smiles to be read and processed (tokenized) at once",
                        default=32)
    parser.add_argument("--num_workers", type=int, required=False,
                        dest="num_workers",
                        help="Number of workers to be used in multiprocessing",
                        default=4)
    parser.add_argument("--enumerations", type=int, required=False,
                        dest="enumerations",
                        help="Number of enumerations to perform on a smile string",
                        default=5)
    args = parser.parse_args()
    assert os.path.exists(args.input)
    return args

def process_data(line, enumerations):
    # First column is zincID and second column is the smiles string
    all_smiles = []
    zinc_id, smiles = line.strip().split(",")
    #Ignore header
    if (zinc_id == "zinc_id") and (smiles == "smiles"): return all_smiles
    #canonicalize the smile string
    mol = Chem.MolFromSmiles(smiles)
    canon_smiles = Chem.MolToSmiles(mol, canonical=True)
    all_smiles.append(canon_smiles)
    atom_order: List[int] = list(range(mol.GetNumAtoms()))
    while(enumerations):
        print ("Enumerations!")
        np.random.shuffle(atom_order)
        aug_mol = Chem.RenumberAtoms(mol, atom_order)
        try:
            aug_smiles = Chem.MolToSmiles(aug_mol, canonical=False)
            if aug_smiles not in all_smiles:
                all_smiles.append(aug_smiles)
        except:
            ## If RDKit couldn't generate augmented smile, we ignore and try again
            pass
        enumerations = enumerations-1
    return all_smiles

def main():
    args = parse_args()

    inputfiles = []
    if os.path.isdir(args.input):
        inputfiles = [ifile for path, subdir, files in os.walk(args.input)
                      for ifile in glob(os.path.join(path, "*.csv"))]
    else:
        if args.input[len(args.input)-3:] != "csv":
            raise ValueError(f'Expected input to be a csv file, but is {args.input[:-3]}')
        inputfiles = [args.input]
    if len(inputfiles) == 0:
        raise FileNotFoundError('No .csv files found in folder.')
    else:
        print(f'Found {len(inputfiles)} .csv files.') 

    #merge default config with the user input config.
    cfg = OmegaConf.load(args.config)
    default_tokenizer = OmegaConf.create(MolEncTokenizerFromVocabFileConfig())
    cfg_tokenizer = cfg.tokenizer if cfg.get('tokenizer', False) else default_tokenizer
    merge_cfg_tokenizer = OmegaConf.merge(default_tokenizer, cfg_tokenizer)
    if not os.path.exists(merge_cfg_tokenizer.vocab_path):
        raise ValueError(f'Vocab file not found at {merge_cfg_tokenizer.vocab_path}')

    # Initialize tokenizer
    tokenizer = MolEncTokenizer.from_vocab_file(**merge_cfg_tokenizer)

    # Initialize a dataset writer
    os.makedirs(args.out_dir, exist_ok = True)
    output_file = os.path.join(args.out_dir, "data.bin")
    index_file = os.path.join(args.out_dir, "data.idx")

    # We do not overwrite the existing data
    if (os.path.exists(output_file) or os.path.exists(index_file)):
        raise FileExistsError(f'The output file at the location {output_file} and {index_file}')
    dataset_builder = indexed_dataset.make_builder(output_file, impl="mmap", vocab_size=tokenizer.vocab_size)
    partial_process_func = partial(process_data, enumerations=args.enumerations)
    # Write the binarized output
    sizes = []
    batched_smiles = []
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(args.num_workers)
    for inputfile in inputfiles:
        ifile = open(inputfile, "r")
        out_iterator = pool.imap(partial_process_func, ifile, 25)
        for smiles in out_iterator:
            token_output = tokenizer.tokenize(smiles, pad=False)
            enc_token_ids = tokenizer.convert_tokens_to_ids(token_output["original_tokens"])
            enc_token_ids = torch.tensor(enc_token_ids)
            dataset_builder.add_item(enc_token_ids)
        dataset_builder.end_document()

    dataset_builder.finalize(index_file)
   

if __name__ == '__main__':
    main()
