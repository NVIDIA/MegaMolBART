# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import os
from copy import deepcopy
from omegaconf.dictconfig import DictConfig
from omegaconf import open_dict

from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_lm_encoder_decoder_model import (
    MegatronLMEncoderDecoderModel,
)
from nemo.utils import logging

from nemo_chem.tokenizer import MolEncTokenizer, DEFAULT_VOCAB_PATH
from nemo_chem.data import MoleculeEnumeration, build_train_valid_test_datasets


__all__ = ["MegaMolBARTModel"]

DATASET_ENUM = ['zinc_csv'] # TODO UPDATE WITH OTHER TYPES

class MegaMolBARTModel(MegatronLMEncoderDecoderModel):
    """
    MegaMolBART pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self._tokenizer_config = cfg.tokenizer  # TODO replace this with get_cheminformatics_tokenizer
        super().__init__(cfg, trainer=trainer)
        self.collate_fn = MoleculeEnumeration(tokenizer=self.tokenizer, seq_length=self._cfg.seq_length, **self._cfg.data).collate_fn # TODO remove when data loader complete

    def _build_tokenizer(self):
        """
        Tokenizer from MegaMolBART.
        """
        vocab_path = self._tokenizer_config.get('vocab_path', DEFAULT_VOCAB_PATH) # TODO replace this with get_cheminformatics_tokenizer
        if not os.path.exists(vocab_path):
            raise ValueError(f'Vocab file not found at {vocab_path}')

        self.tokenizer = MolEncTokenizer.from_vocab_file(vocab_path=vocab_path, **self._tokenizer_config)

    def _build_vocab(self):
        """
        Manipulate vocabulary (e.g., pad vocabulary for increased performance)/
        """
        # TODO: add to config to allow this to be disabled?
        self.padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=len(self.tokenizer),
            make_vocab_size_divisible_by=self._cfg.get('make_vocab_size_divisible_by', 128),
            tensor_model_parallel_size=self._cfg.get('tensor_model_parallel_size', 1),
        )

    def build_train_valid_test_datasets(self):
        logging.info('Building MegaMolBART datasets.')
        tensor_model_parallel_size = self._cfg.get('tensor_model_parallel_size', 1)

        global_batch_size = self.trainer.world_size * self._cfg.micro_batch_size / tensor_model_parallel_size
        eval_iters = (self.trainer.max_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches
        train_valid_test_num_samples = [
            int(self.trainer.max_steps * global_batch_size),
            int(eval_iters * global_batch_size),
            int(test_iters * global_batch_size),
        ]

        # Make sure the user specifies dataset type as 'megamolbart_csv' only.
        if self._cfg.data.get('dataset_type', None) is not None:
            if self._cfg.data.get('dataset_type') not in DATASET_ENUM:
                raise ValueError(f"dataset_type must be in {DATASET_ENUM}. Found {self._cfg.data.get('dataset_type')}")

        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            self._cfg.data,
            self.trainer,
            train_valid_test_num_samples
        )

        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building MegaMolBART datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""
        dataloader = super().build_pretraining_data_loader(dataset=dataset, consumed_samples=consumed_samples)

        if dataloader is not None:
            dataloader.collate_fn = self.collate_fn
        
        return dataloader

    def list_available_models(self):
        pass
