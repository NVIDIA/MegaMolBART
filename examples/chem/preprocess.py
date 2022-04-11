# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

import os
import sys
import requests
import multiprocessing as mp
import pandas as pd

from datetime import datetime
from subprocess import run
from multiprocessing import Pool
from functools import partial

from rdkit import Chem

from nemo.utils import logging


MAX_LENGTH = 150


class Preprocess(object):

    def __init__(self) -> None:
        super().__init__()
        self.retry = False

    def _run_cmd(self, cmd, failure_error='Unexpected error while executing bash cmd'):
        logging.debug(f'Running cmd: {cmd}')

        process = run(['bash', '-c', cmd], capture_output=True, text=True)

        if process.returncode != 0:
            logging.error(failure_error)
            sys.exit(process.returncode)
        return process

    def _process_file(self, url, download_dir='/tmp/zinc15/raw'):

        filename = url.split('/')[-1]
        if os.path.exists(os.path.join(download_dir, filename)):
            logging.info(f'{url} already downloaded...')
            return

        logging.debug(f'Downloading file {filename}...')
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                tmp_filename = os.path.join(download_dir, filename + '_tmp')
                header = True
                with open(tmp_filename, 'w') as f:
                    for line in r.iter_lines():
                        if header:
                            header = False
                            f.write("zinc_id,SMILES\n")
                            continue
                        line = line.decode("utf-8")
                        splits = line.split("\t")
                        if len(splits) < 2:
                            continue

                        smi, zinc_id = splits[0], splits[1]
                        try:
                            mol = Chem.MolFromSmiles(smi)
                            smi = Chem.MolToSmiles(mol, canonical=True)
                        except RuntimeError:
                            continue

                        if len(smi) > MAX_LENGTH:
                            continue

                        f.write(f"{zinc_id},{smi}\n")
            os.rename(tmp_filename, os.path.join(download_dir, filename))
            return
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logging.error(f'{url} Not found')
                return
            else:
                logging.error(
                    f'Could not download file {url}: {e.response.status_code}')
                raise e

    def __processing_failure(self, e):
        logging.info(f'Processing failure: {e}')
        self.retry = True

    def process_files(self, links_file, pool_size=8, download_dir='/tmp/zinc15/raw'):
        """
        Download all the files in the links file.

        Parameters:
            links_file (str): File containing links to be downloaded.
            pool_size (int): Number of processes to use.
            download_dir (str): Directory to download the files to.
        """

        logging.info(
            f'Downloading files from {links_file} with poolsize {pool_size}...')

        os.makedirs(download_dir, exist_ok=True)
        with open(links_file, 'r') as f:
            links = list(set([x.strip() for x in f]))

        download_funct = partial(self._process_file, download_dir=download_dir)

        while True:
            pool = Pool(processes=pool_size)
            pool.map_async(download_funct,
                           links,
                           error_callback=self.__processing_failure)
            pool.close()
            pool.join()

            if self.retry:
                logging.info(
                    'Retrying to download files that failed with 503...')
                self.retry = False
            else:
                break

    def _process_split(self, datafile, output_dir='/tmp/zinc15/processed/'):
        filename = f'{output_dir}/split_data/{datafile}'
        logging.info(f'Processing file {filename}...')

        df = pd.read_csv(filename, header=None, names=['zinc_id', 'smiles'])
        recs = int(df.shape[0] * 0.01)

        test_df = df.sample(n=recs)
        df = df.drop(test_df.index)  # remove test data from training data

        val_df = df.sample(n=recs)
        df = df.drop(val_df.index)  # remove test data from training data

        df.to_csv(f'{output_dir}/train/{datafile}.csv', index=False)
        test_df.to_csv(f'{output_dir}/test/{datafile}.csv', index=False)
        val_df.to_csv(f'{output_dir}/val/{datafile}.csv', index=False)

        with open(f'{output_dir}/val/metadata.csv', 'a') as f:
            f.write(f"{datafile},{val_df.shape[0]}\n")
        with open(f'{output_dir}/test/metadata.csv', 'a') as f:
            f.write(f"{datafile},{test_df.shape[0]}\n")
        with open(f'{output_dir}/train/metadata.csv', 'a') as f:
            f.write(f"{datafile},{df.shape[0]}\n")

        del df
        del test_df
        del val_df

    def balanced_split(self, download_dir, output_dir, pool_size=8):
        if os.path.exists(output_dir):
            logging.info(f'{output_dir} already exists...')
            os.rename(output_dir, output_dir +
                      datetime.now().strftime('%Y%m%d%H%M%S'))

        split_data = os.path.join(output_dir, 'split_data')
        os.makedirs(split_data, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)

        self._run_cmd(f"cd {split_data}; tail -q -n +2 {download_dir}/** | split -d -l 10000000 -a 3",
                      failure_error='Error while merging files')

        split_files = os.listdir(split_data)

        with open(f'{output_dir}/val/metadata.csv', 'w') as f:
            f.write(f"file,size\n")
        with open(f'{output_dir}/test/metadata.csv', 'w') as f:
            f.write(f"file,size\n")
        with open(f'{output_dir}/train/metadata.csv', 'w') as f:
            f.write(f"file,size\n")

        with Pool(processes=pool_size) as pool:
            split_funct = partial(self._process_split, output_dir=output_dir)

            pool.map(split_funct,
                     split_files)

    def prepare_dataset(self,
                        links_file='conf/model/dataset/ZINC-downloader-AZ.txt',
                        download_dir='/tmp/zinc15/raw',
                        output_dir='/tmp/zinc15/processed'):
        """
        Download zinc15 dataset and slits it into train, valid, and test sets.

        Parameters:
            links_file (str): File containing links to be downloaded.
            download_dir (str): Directory to download the files to.
            output_dir (str): Directory to save the processed data to.
        """
        # More than 8 cores may cause 503 errors. Please avoid larger pool size.
        self.process_files(links_file,
                           pool_size=8,
                           download_dir=download_dir)
        logging.info('Download complete.')

        self.balanced_split(download_dir,
                            output_dir,
                            pool_size=8)
