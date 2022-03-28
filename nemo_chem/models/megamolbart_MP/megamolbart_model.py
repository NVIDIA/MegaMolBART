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

from nemo.collections.nlp.models.language_modeling.megatron_lm_encoder_decoder_model import MegatronLMEncoderDecoderModel
from nemo.utils import logging

from nemo_chem.tokenizer import MolEncTokenizer, DEFAULT_VOCAB_PATH
from nemo_chem.data import DatasetTypes, MoleculeEnumeration, build_train_valid_test_datasets
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group

# Disable logging of invalid SMILES moloecules
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

__all__ = ["MegaMolBARTModel"]

class MegaMolBARTModel(MegatronLMEncoderDecoderModel):
    """
    MegaMolBART pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self._tokenizer_config = cfg.tokenizer  # TODO replace this with get_cheminformatics_tokenizer
        super().__init__(cfg, trainer=trainer)

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

        if self._cfg.data.get('dataset_type', None) is not None:
            dataset_types = DatasetTypes.__members__
            if self._cfg.data.get('dataset_type') not in dataset_types:
                raise ValueError(f"dataset_type must be in {dataset_types}. Found {self._cfg.data.get('dataset_type')}")

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
        
        # Add collate function and unpin memory to avoid crash with CUDA misaligned address
        # TODO remove when data loader complete
        dataloader.pin_memory = False
        dataloader.collate_fn = MoleculeEnumeration(tokenizer=self.tokenizer, seq_length=self._cfg.seq_length, **self._cfg.data).collate_fn
        
        return dataloader

    def _eval_step(self, tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask):
        ret_dict = self(tokens_enc, tokens_dec, enc_mask, dec_mask, tokentype_ids=None, lm_labels=labels,)
        tokens_loss = ret_dict['tokens_loss']
        loss = self.loss_func(loss_mask, tokens_loss)
        return loss, ret_dict

    def training_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask = self.process_batch(batch)

        assert tokens_enc.max() < self.tokenizer.vocab_size, AssertionError('Encoder tokens are larger than vocabulary')
        assert tokens_dec.max() < self.tokenizer.vocab_size, AssertionError('Decoder tokens are larger than vocabulary')
        assert labels.max() < self.tokenizer.vocab_size, AssertionError('Label tokens are larger than vocabulary')

        loss, ret_dict = self._eval_step(tokens_enc=tokens_enc, tokens_dec=tokens_dec, loss_mask=loss_mask, 
                                         labels=labels, enc_mask=enc_mask, dec_mask=dec_mask)
        
        # cache reduced loss while accumulating gradients
        reduced_loss = average_losses_across_data_parallel_group([loss])
        self._reduced_loss_buffer.append(reduced_loss[0])

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            self.log('global_step', self.trainer.global_step, prog_bar=True)

            # Reduced loss for logging.
            average_reduced_loss = sum(self._reduced_loss_buffer) / len(self._reduced_loss_buffer)
            self.log('reduced_loss', average_reduced_loss, prog_bar=True)
            
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr)

            consumed_samples = self.compute_consumed_samples(self.trainer.global_step - self.init_global_step)            
            self.log('consumed_samples', consumed_samples, prog_bar=True)

            self._reduced_loss_buffer = []

        return loss

    def validation_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask = self.process_batch(batch)
        loss, ret_dict = self._eval_step(tokens_enc=tokens_enc, tokens_dec=tokens_dec, loss_mask=loss_mask, 
                                         labels=labels, enc_mask=enc_mask, dec_mask=dec_mask)

        self.log('global_step', self.trainer.global_step, prog_bar=True)

        reduced_loss = average_losses_across_data_parallel_group([loss])
        self.log('reduced_loss', reduced_loss, prog_bar=True)

        target_smiles = batch['target_smiles']
        token_logits = ret_dict['token_logits']
        metrics = self.calculate_metrics(token_logits, loss_mask, labels, tokens_enc, enc_mask, target_smiles)
        for metric_name in metrics:
            self.log(f'{metric_name}', metrics[metric_name], prog_bar=True)

        return reduced_loss

    @staticmethod
    def _calculate_character_accuracy(token_logits, loss_mask, labels):
        """Character (token) level accuracy

        Args:
            token_logits (torch.Tensor, float): softmax values for all tokens
            loss_mask (torch.Tensor, float): binary mask for ignored data (1=active, 0=mask), must be float
            labels (torch.Tensor, long): token IDs for correct output

        Returns:
            float: character accuracy value
        """

        # Get most likely token
        _, predicted_tokens = torch.max(token_logits, dim=2)
        correct_tokens = torch.eq(labels, predicted_tokens) * loss_mask

        # Calculate percent of correct tokens
        num_correct = correct_tokens.sum().cpu().detach().item()
        total = loss_mask.sum().cpu().detach().item()
        character_accuracy = num_correct / total
        return character_accuracy

    def _sample_molecules(self, tokens_enc, enc_mask):
        """Sample SMILES molecules from encoder hidden state

        Args:
            tokens_enc (torch.Tensor, long): token ID values for samples
            enc_mask (torch.Tensor, long): boolean mask for padded sections

        Returns:
            string: sampled SMILE values
        """

        # Decode encoder hidden state to tokens
        predicted_tokens_dec, log_probs = self.decode(tokens_enc, enc_mask, self._cfg.max_position_embeddings)
        tokens = self.tokenizer.convert_ids_to_tokens(predicted_tokens_dec.tolist())

        # Convert to SMILES string
        sampled_smiles = self.tokenizer.detokenize(tokens)
        return sampled_smiles

    def _calculate_molecular_accuracy(self, tokens_enc, enc_mask, target_smiles):
        """Calculate molecular accuracy (with canonicalization)

        Args:
            tokens_enc (torch.Tensor, long): token ID values for samples
            enc_mask (torch.Tensor, long): boolean mask for padded sections
            target_smiles (str): ground truth for canonicalized SMILES

        Returns:
            float, float: molecular accuracy and percent invalid
        """

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
        """Calculate metrics for character accuracy, molecular accuracy, and invalid molecules

        Args:
            token_logits (torch.Tensor, float): softmax values for all tokens
            loss_mask (torch.Tensor, float): binary mask for ignored data (1=active, 0=mask), must be float
            labels (torch.Tensor, long): token IDs for correct output
            tokens_enc (torch.Tensor, long): token ID values for samples
            enc_mask (torch.Tensor, long): boolean mask for padded sections
            target_smiles (str): ground truth for canonicalized SMILES

        Returns:
            dict: dictionary of metric values
        """
        character_accuracy = self._calculate_character_accuracy(token_logits, loss_mask, labels)
        molecular_accuracy, percent_invalid = self._calculate_molecular_accuracy(tokens_enc, enc_mask, target_smiles)
        metrics = {'character_accuracy': character_accuracy,
                   'molecular_accuracy': molecular_accuracy,
                   'percent_invalid': percent_invalid}
        return metrics

    def list_available_models(self):
        pass
