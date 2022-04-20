
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"Preprocess megamolbart data"
"Take text files as input and output tokenized data in binarized format"
##TODO Merge it with the CSV format data preprocessing script

import argparse
import gzip
import json
import os
import sys
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
import re

DATAFORMAT_EXT = [".csv", ".CSV"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        dest="input",
                        help="Path to a folder where input files \
                        exist. We recursively go through and \
                        convert all CSV files into binarized output. \
                        The output folder structure will be identical to\
                        the input structure. There is one .bin and one .idx \
                        file corresponding to each input CSV.")
    parser.add_argument("--config", type=str, required=True,
                        dest="config",
                        help="The training config file path. Tokenizer params will be\
                              extracted and used for preparing the data.")
    parser.add_argument("--out-dir", type=str, required=True,
                        dest="out_dir",
                        help="Path where the output is to be stored. This should not be a file name,\
                        just a directory where the output is to be saved.",
                        default=".")
    parser.add_argument("--num-workers", type=int, required=False,
                        dest="num_workers",
                        help="Number of workers to be used in multiprocessing",
                        default=4)
    parser.add_argument("--num_enumerations", type=int, required=False,
                        dest="num_enumerations",
                        help="Number of enumerations to perform on a smile string",
                        default=5)
    args = parser.parse_args()
    assert os.path.exists(args.input), "Given input path %s does not exist, please provide a valid path" % args.input 
    return args

def process_data(line, num_enumerations, tokenizer):
    # First column is zincID and second column is the smiles string
    all_smiles = []
    #Ignore header
    if not (re.search(r"""ZINC[0-9]{12}""", line)): return all_smiles
    zinc_id, smiles = line.strip().split(",")
    #canonicalize the smile string
    mol = Chem.MolFromSmiles(smiles)
    canon_smiles = Chem.MolToSmiles(mol, canonical=True)
    all_smiles.append(canon_smiles)
    atom_order: List[int] = list(range(mol.GetNumAtoms()))
    while(num_enumerations):
        np.random.shuffle(atom_order)
        aug_mol = Chem.RenumberAtoms(mol, atom_order)
        try:
            aug_smiles = Chem.MolToSmiles(aug_mol, canonical=False)
            if aug_smiles not in all_smiles:
                all_smiles.append(aug_smiles)
        except:
            ## If RDKit couldn't generate augmented smile, we ignore and try again
            pass
        num_enumerations = num_enumerations-1
    token_output = tokenizer.tokenize(all_smiles, pad=False)
    enc_token_ids = tokenizer.convert_tokens_to_ids(token_output["original_tokens"])
    return enc_token_ids

def initialize_tokenizer(inputconfig):
    cfg = OmegaConf.load(inputconfig)
    default_tokenizer = OmegaConf.create(MolEncTokenizerFromVocabFileConfig())
    cfg_tokenizer = cfg.tokenizer if cfg.get('tokenizer', False) else default_tokenizer
    merge_cfg_tokenizer = OmegaConf.merge(default_tokenizer, cfg_tokenizer)
    if not os.path.exists(merge_cfg_tokenizer.vocab_path):
        raise ValueError(f'Vocab file not found at {merge_cfg_tokenizer.vocab_path}')

    # Initialize tokenizer
    tokenizer = MolEncTokenizer.from_vocab_file(**merge_cfg_tokenizer)

    return tokenizer

def main():
    args = parse_args()

    inputfiles = []
    assert os.path.isdir(args.input), "Expected --input to be a directory."
    inputfiles = [ifile for path, subdir, files in os.walk(args.input)
                  for dformat in DATAFORMAT_EXT
                  for ifile in glob(os.path.join(path, "*" + dformat))]
    if len(inputfiles) == 0:
        raise FileNotFoundError('No CSV files found in folder.')
    else:
        print(f'Found {len(inputfiles)} .csv files.') 

    ## Make sure there are no bin, idx files inside the output directory
    # Initialize a dataset writer
    os.makedirs(args.out_dir, exist_ok = True)
    os.access(args.out_dir, os.W_OK)
    outbinfiles = []
    for path, subdir, files in os.walk(args.out_dir):
        outbinfiles = [ifile for path, subdir, files in os.walk(args.out_dir)
                      for dformat in DATAFORMAT_EXT
                      for ifile in glob(os.path.join(path, "*.bin"))]

    assert len(outbinfiles) == 0, "Found some .bin files at the output location %s."
    "Cannot overwrite the existing data. Please delete or pass a new output location." % outbinfiles
    tokenizer = initialize_tokenizer(args.config)
    
    # Create an identical folder structure in the output directory as the input dir.
    for path, subdir, files in os.walk(args.input):
        newpath = os.path.join(args.out_dir, path[len(args.input):])
        if not os.path.isdir(newpath):
            os.mkdir(newpath)

    partial_process_func = partial(process_data, num_enumerations=args.num_enumerations, tokenizer=tokenizer)
    # Write the binarized output
    sizes = []
    batched_smiles = []
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(args.num_workers)
    for inputfile in inputfiles:
        subfolder_path = os.path.dirname(inputfile[len(args.input)+1:])
        print(subfolder_path)
        ifilebasename = os.path.splitext(os.path.basename(inputfile))[0]
        print(ifilebasename)
        output_file = os.path.join(args.out_dir, subfolder_path, ifilebasename + ".bin")
        print(output_file)
        index_file = os.path.join(args.out_dir, subfolder_path, ifilebasename + ".idx")

        dataset_builder = indexed_dataset.make_builder(output_file, impl="mmap", vocab_size=tokenizer.vocab_size)
        ifile = open(inputfile, "r")
        out_iterator = pool.imap(partial_process_func, ifile, 25)
        for enc_token_ids in out_iterator:
            enc_token_ids = torch.tensor(enc_token_ids)
            dataset_builder.add_item(enc_token_ids)
        dataset_builder.end_document()
        dataset_builder.finalize(index_file)
   

if __name__ == '__main__':
    main()
