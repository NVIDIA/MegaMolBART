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
from rdkit import Chem

import torch
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_lm_encoder_decoder_model import (
    MegatronLMEncoderDecoderModel,
)
from nemo.utils import logging

from nemo_chem.tokenizer import MolEncTokenizer, DEFAULT_VOCAB_PATH
from nemo_chem.data import MoleculeEnumeration, build_train_valid_test_datasets
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group

# Disable logging of invalid SMILES moloecules
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

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
        # self._val_metrics_inputs = defaultdict(list)

    def _build_tokenizer(self):
        """
        Tokenizer from MegaMolBART.
        """
        vocab_path = self._tokenizer_config.get('vocab_path', DEFAULT_VOCAB_PATH) # TODO replace this with get_cheminformatics_tokenizer
        if not os.path.exists(vocab_path):
            raise ValueError(f'Vocab file not found at {vocab_path}')

        self.tokenizer = MolEncTokenizer.from_vocab_file(vocab_path=vocab_path, **self._tokenizer_config)

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

    def process_batch(self, batch):
        """Custom batch processing to append SMILES strings for metrics"""
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask = super().process_batch(batch)
        return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, batch['target_smiles']

    def _eval_step(self, tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask):
        ret_dict = self(tokens_enc, tokens_dec, enc_mask, dec_mask, tokentype_ids=None, lm_labels=labels,)
        tokens_loss = ret_dict['tokens_loss']
        loss = self.loss_func(loss_mask, tokens_loss)
        return loss, ret_dict

    def training_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, target_smiles = self.process_batch(batch)
        loss, ret_dict = self._eval_step(tokens_enc=tokens_enc, tokens_dec=tokens_dec, loss_mask=loss_mask, 
                                         labels=labels, enc_mask=enc_mask, dec_mask=dec_mask)
        
        # cache reduced loss while accumulating gradients
        reduced_loss = average_losses_across_data_parallel_group([loss])
        self._reduced_loss_buffer.append(reduced_loss[0])

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            self.log('global_step', self.trainer.global_step, prog_bar=True)

            # Reduced loss for logging.
            average_reduced_loss = sum(self._reduced_loss_buffer) / len(self._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True)
            
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr)

            consumed_samples = self.compute_consumed_samples(self.trainer.global_step - self.init_global_step)            
            self.log('consumed_samples', consumed_samples, prog_bar=True)

            self._reduced_loss_buffer = []

        return loss

    def validation_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, target_smiles = self.process_batch(batch)
        loss, ret_dict = self._eval_step(tokens_enc=tokens_enc, tokens_dec=tokens_dec, loss_mask=loss_mask, 
                                         labels=labels, enc_mask=enc_mask, dec_mask=dec_mask)

        self.log('global_step', self.trainer.global_step, prog_bar=True)

        reduced_loss = average_losses_across_data_parallel_group([loss])
        self.log('reduced_loss', reduced_loss, prog_bar=True)

        token_logits = ret_dict['token_logits']
        metrics = self.calculate_metrics(token_logits, loss_mask, labels, tokens_enc, enc_mask, target_smiles)
        for metric_name in metrics:
            self.log(metric_name, metrics[metric_name], prog_bar=False)

        return reduced_loss

    @staticmethod
    def _calculate_character_accuracy(token_logits, loss_mask, labels):
        """Character (token) level accuracy"""
        _, predicted_tokens = torch.max(token_logits, dim=2)
        correct_tokens = torch.eq(labels, predicted_tokens) * loss_mask
        num_correct = correct_tokens.sum().cpu().detach().item()
        total = loss_mask.sum().cpu().detach().item()
        character_accuracy = num_correct / total
        return character_accuracy

    def _sample_molecules(self, tokens_enc, enc_mask):
        predicted_tokens_dec, log_probs = self.decode(tokens_enc, enc_mask, self._cfg.max_position_embeddings)
        tokens = self.tokenizer.convert_ids_to_tokens(predicted_tokens_dec.tolist())
        sampled_smiles = self.tokenizer.detokenize(tokens)
        return sampled_smiles

    def _calculate_molecular_accuracy(self, tokens_enc, enc_mask, target_smiles):
        sampled_smiles = self._sample_molecules(tokens_enc, enc_mask)
        sampled_mols = [Chem.MolFromSmiles(smi) for smi in sampled_smiles]
        invalid = [mol is None for mol in sampled_mols]

        canonical_smiles = ["Unknown" if mol is None else Chem.MolToSmiles(mol, canonical=True) for mol in sampled_mols]
        correct_smiles = [target_smiles[idx] == smi for idx, smi in enumerate(canonical_smiles)]

        num_correct = sum(correct_smiles)
        total = len(correct_smiles)
        num_invalid = sum(invalid)
        percent_invalid = num_invalid / total
        molecular_accuracy = num_correct / total

        return molecular_accuracy, percent_invalid

    def calculate_metrics(self, token_logits, loss_mask, labels, tokens_enc, enc_mask, target_smiles):
        character_accuracy = self._calculate_character_accuracy(token_logits, loss_mask, labels)
        molecular_accuracy, percent_invalid = self._calculate_molecular_accuracy(tokens_enc, enc_mask, target_smiles)
        metrics = {'character_accuracy': character_accuracy,
                   'molecular_accuracy': molecular_accuracy,
                   'percent_invalid': percent_invalid}
        return metrics

    def list_available_models(self):
        pass
