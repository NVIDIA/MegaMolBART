
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

__all__ = ['CsvToBinary']

class CsvToBinary:
    def __init__(self, input_dir: str, out_dir: str,
                 config, num_enumerations, num_workers):
        """
        input_dir: This is the directory where the CSV data exists.
        We support nested directory structure for input directories
        out_dir: Directory path to save the bin files. out_dir will mimic
        the same directory structure as input_dir and the output files
        will be named identical to the input files, but with a .bin extension
        num_enumerations: Number of enumerations to perform on every smile
        num_workers: Number of workers to use for multi-processing.
        """
        self.input_dir = input_dir
        self.out_dir = out_dir
        self.cfg = config
        self.num_enumerations = num_enumerations
        self.num_workers = num_workers
        self.tokenizer = self._initialize_tokenizer()      

        assert os.path.isdir(input_dir), "Expected --input to be a directory."
        self.inputfiles = [ifile for path, subdir, files in os.walk(self.input_dir)
                      for dformat in DATAFORMAT_EXT
                      for ifile in glob(os.path.join(path, "*" + dformat))]
        if len(self.inputfiles) == 0:
            raise FileNotFoundError('No CSV files found in folder.')
        else:
            print(f'Found {len(self.inputfiles)} .csv files.') 

        # If the destination path is not the same as where the CSVs exist, make an identical
        # folder structure as the input directory at the destination
        if self.out_dir != self.input_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            os.access(self.out_dir, os.W_OK)
            # Create an identical folder structure in the output directory as the input dir.
            for path, subdir, files in os.walk(self.input_dir):
                subdir = path[len(self.input_dir)+1:]
                folder_path = os.path.join(self.out_dir, subdir)
                os.makedirs(os.path.join(self.out_dir, subdir), exist_ok=True)

        self.outbinfiles = []
        self.outidxfiles = []
        for path, subdir, files in os.walk(self.out_dir):
            outbinfiles = [ifile for path, subdir, files in os.walk(self.out_dir)
                          for dformat in DATAFORMAT_EXT
                          for ifile in glob(os.path.join(path, "*.bin"))]
            assert len(self.outbinfiles) == 0, "Found existing .bin files at the output location %s."
            "Cannot overwrite the existing data. Please delete and retry." % outbinfiles
            outidxfiles = [ifile for path, subdir, files in os.walk(self.out_dir)
                          for dformat in DATAFORMAT_EXT
                          for ifile in glob(os.path.join(path, "*.bin"))]
            assert len(self.outidxfiles) == 0, "Found existing .idx files at the output location %s."
            "Cannot overwrite the existing data. Please delete and retry." % outidxfiles


    def _initialize_tokenizer(self):
        default_tokenizer = OmegaConf.create(MolEncTokenizerFromVocabFileConfig())
        cfg_tokenizer = self.cfg.tokenizer if self.cfg.get('tokenizer', False) else default_tokenizer
        merge_cfg_tokenizer = OmegaConf.merge(default_tokenizer, cfg_tokenizer)
        if not os.path.exists(merge_cfg_tokenizer.vocab_path):
            raise ValueError(f'Vocab file not found at {merge_cfg_tokenizer.vocab_path}')

        # Initialize tokenizer
        tokenizer = MolEncTokenizer.from_vocab_file(**merge_cfg_tokenizer)

        return tokenizer

    def prepare_dataset(self):
        sizes = []
        batched_smiles = []
        pool = multiprocessing.Pool(self.num_workers)
        for inputfile in self.inputfiles:
            #ignore metadata.csv files
            if "metadata.csv" in inputfile: continue
            subfolder_path = os.path.dirname(inputfile[len(self.input_dir)+1:])
            ifilebasename = os.path.splitext(os.path.basename(inputfile))[0]
            output_file = os.path.join(self.out_dir, subfolder_path, ifilebasename + ".bin")
            index_file = os.path.join(self.out_dir, subfolder_path, ifilebasename + ".idx")

            dataset_builder = indexed_dataset.make_builder(output_file, impl="mmap", vocab_size=self.tokenizer.vocab_size)
            ifile = open(inputfile, "r")
            out_iterator = pool.imap(self._process_data, ifile, 25)
            for enc_token_ids in out_iterator:
                #we may return an empty list when the row doesn't match with our regex query
                if not enc_token_ids: continue
                ## If num_enumerations > 0, we will have more than one element
                #in the list and we can't convert the list of lists into torch
                #tensor because they all may have different lengths.
                #Padding should only be done during training, so we cannot pad them here.
                for enc_token_id in enc_token_ids:
                    dataset_builder.add_item(torch.tensor(enc_token_id))
            dataset_builder.end_document()
            dataset_builder.finalize(index_file)

    def _process_data(self, line):
        # First column is zincID and second column is the smiles string
        all_smiles = []
        #Ignore header
        ##TODO: This is specific to the current AZ format. Make this a config param in future.
        if not (re.match('^[0-9]+', line)): return all_smiles
        zinc_id, smiles = line.strip().split(",")
        all_smiles.append(smiles)
        mol = Chem.MolFromSmiles(smiles)
        atom_order: List[int] = list(range(mol.GetNumAtoms()))
        while(self.num_enumerations):
            np.random.shuffle(atom_order)
            aug_mol = Chem.RenumberAtoms(mol, atom_order)
            try:
                aug_smiles = Chem.MolToSmiles(aug_mol, canonical=False)
                if aug_smiles not in all_smiles:
                    all_smiles.append(aug_smiles)
            except:
                ## If RDKit couldn't generate augmented smile, we ignore and try again
                pass
            self.num_enumerations -= 1
        token_output = self.tokenizer.tokenize(all_smiles, pad=False)
        enc_token_ids = self.tokenizer.convert_tokens_to_ids(token_output["original_tokens"])
        return enc_token_ids
