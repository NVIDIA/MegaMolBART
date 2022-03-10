import os
import re
from operator import itemgetter
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig
import torch
from pytorch_lightning.trainer.trainer import Trainer

from nemo.utils import AppState, logging
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)

from nemo.collections.nlp.models.language_modeling.megatron_lm_encoder_decoder_model import MegatronLMEncoderDecoderModel
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group

from nemo_chem.tokenizer import MolEncTokenizer, DEFAULT_VOCAB_PATH
from nemo_chem.data import MoleculeEnumeration, build_train_valid_test_datasets

try:
    from apex.transformer import parallel_state, tensor_parallel

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

__all__ = ["MegaMolBARTModel"]


class MegaMolBARTModel(MegatronLMEncoderDecoderModel):
    """
    MegaMolBART model
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self._tokenizer_config = cfg.tokenizer  # TODO replace this with get_cheminformatics_tokenizer
        super().__init__(cfg.model, trainer=trainer)

        # TODO collate functions are not currently threadsafe and have to be declared in itit
        self.train_collate = MoleculeEnumeration(tokenizer=self.tokenizer, seq_length=self._cfg.seq_len, **self._cfg.train_dataset)
        self.valid_collate = MoleculeEnumeration(tokenizer=self.tokenizer, seq_length=self._cfg.seq_len, **self._cfg.validation_dataset)
        self.test_collate = MoleculeEnumeration(tokenizer=self.tokenizer, seq_length=self._cfg.seq_len, **self._cfg.test_dataset)

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
        # TODO: add config to allow to disable it?
        self.padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=len(self.tokenizer),
            make_vocab_size_divisible_by=self._cfg.get('make_vocab_size_divisible_by', 128),
            tensor_model_parallel_size=self._cfg.get('tensor_model_parallel_size', 1),
        )

    def calculate_train_val_test_num_samples(self, tensor_model_parallel_size: int):
        # TODO FIX SO COMPATIBLE WITH EPOCHS AND CHECK VAL_CHECK_INTERVAL CALC
        train_global_batch_size = self.trainer.world_size * self._cfg.train_dataset.micro_batch_size / tensor_model_parallel_size

        if self._cfg.get('validation_dataset'):
            valid_global_batch_size = self.trainer.world_size * self._cfg.validation_dataset.micro_batch_size / tensor_model_parallel_size
        else:
            valid_global_batch_size = 0

        if self._cfg.get('test_dataset'):
            test_global_batch_size = self.trainer.world_size * self._cfg.test_dataset.micro_batch_size / tensor_model_parallel_size
        else:
            test_global_batch_size = 0

        val_iters = (self.trainer.max_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches
        train_valid_test_num_samples = [
            int(self.trainer.max_steps * train_global_batch_size),
            int(val_iters * valid_global_batch_size),
            int(test_iters * test_global_batch_size),
        ]
        return train_valid_test_num_samples

    def build_train_valid_test_datasets(self):
        cfg = self._cfg.copy()

        logging.info('Building MegaMolBART datasets.')
        tensor_model_parallel_size = self._cfg.get('tensor_model_parallel_size', 1)
        train_valid_test_num_samples = self.calculate_train_val_test_num_samples(tensor_model_parallel_size)
        with open_dict(cfg):
            cfg.train_dataset['num_samples'] = train_valid_test_num_samples[0]
            cfg.train_dataset['tensor_model_parallel_size'] = tensor_model_parallel_size

            if cfg.validation_dataset:
                cfg.validation_dataset['num_samples'] = train_valid_test_num_samples[1]
                cfg.validation_dataset['tensor_model_parallel_size'] = tensor_model_parallel_size

            if cfg.test_dataset:
                cfg.test_dataset['num_samples'] = train_valid_test_num_samples[2]
                cfg.validation_dataset['tensor_model_parallel_size'] = tensor_model_parallel_size

        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            cfg,
            self.trainer
        )
        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building MegaMolBART datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def setup(self, stage=None):
        """A PTL method to setup the training, validation and test datasets."""
        if stage == 'predict':
            return
        if self._train_dl is not None and self._validation_dl is not None:
            return
        self.build_train_valid_test_datasets()
        self.setup_training_data(self._cfg.train_dataset)

        if self._cfg.get('validation_dataset', False):
            self.setup_validation_data(self._cfg.validation_dataset)

        if self._cfg.get('test_dataset', False):
            self.setup_test_data(self._cfg.test_dataset)

    def setup_training_data(self, cfg):
        # TODO send to base class
        if hasattr(self, '_train_ds'):
            resume_checkpoint_path = self.trainer.checkpoint_connector.resume_checkpoint_path
            if resume_checkpoint_path:
                consumed_samples = int(
                    float(re.findall(r"consumed_samples\=([0-9]+.[0-9]+)", resume_checkpoint_path)[0])
                )
            else:
                consumed_samples = 0

            collate_fn = self.train_collate.collate_fn
            self._train_dl = self.build_pretraining_data_loader(cfg, self._train_ds, consumed_samples, collate_fn)

    def setup_validation_data(self, cfg):
        # TODO send to base class
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            collate_fn = self.valid_collate.collate_fn
            self._validation_dl = self.build_pretraining_data_loader(cfg, self._validation_ds, consumed_samples, collate_fn)

    def setup_test_data(self, cfg):
        # TODO send to base class
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            collate_fn = self.test_collate.collate_fn
            self._test_dl = self.build_pretraining_data_loader(cfg, self._test_ds, consumed_samples, collate_fn)

    def compute_consumed_samples(self, global_step, cfg):
        # TODO send to base class
        app_state = AppState()
        consumed_samples = (
            global_step
            * app_state.data_parallel_size
            * cfg.micro_batch_size
            * self.trainer.accumulate_grad_batches
        )
        return int(consumed_samples)

    def build_pretraining_data_loader(self, cfg, dataset, consumed_samples, collate_fn):
        # TODO send to base class
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        # Megatron sampler
        if cfg.dataloader_type == 'single':
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=cfg.micro_batch_size,
                data_parallel_rank=parallel_state.get_data_parallel_rank(),
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        elif cfg.dataloader_type == 'cyclic':
            batch_sampler = MegatronPretrainingRandomSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=cfg.micro_batch_size,
                data_parallel_rank=parallel_state.get_data_parallel_rank(),
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        else:
            raise Exception(f'Dataloader type {cfg.dataloader_type} is not supported.')

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, collate_fn=collate_fn
        )

    def training_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, target_smiles = self.process_batch(batch)

        tokens_loss = itemgetter("tokens_loss")(
            self(tokens_enc, tokens_dec, enc_mask, dec_mask, tokentype_ids=None, lm_labels=labels,)
        )
        loss = self.loss_func(loss_mask, tokens_loss)
        reduced_loss = average_losses_across_data_parallel_group([loss])

        self.log('train_loss', loss)
        # cache reduced loss while accumulating gradients
        self._reduced_loss_buffer.append(reduced_loss[0])

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            average_reduced_loss = sum(self._reduced_loss_buffer) / len(self._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr)
            self.log('global_step', self.trainer.global_step, prog_bar=True)
            self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step, self._cfg.train_dataset), prog_bar=True)
            self._reduced_loss_buffer = []
            if self.cfg.precision == 16:
                loss_scale = self.trainer.precision_plugin.scaler._scale
                if loss_scale is not None:
                    self.log('loss_scale', loss_scale)

    def validation_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, target_smiles = self.process_batch(batch)

        tokens_loss = itemgetter("tokens_loss")(
            self(tokens_enc, tokens_dec, enc_mask, dec_mask, tokentype_ids=None, lm_labels=labels,)
        )
        loss = self.loss_func(loss_mask, tokens_loss)
        reduced_loss = average_losses_across_data_parallel_group([loss])
        return reduced_loss

    def validation_epoch_end(self, outputs):
        # TODO send to base class
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        self.log('val_loss', averaged_loss[0], prog_bar=True)
        self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step, self._cfg.validation_dataset))

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    # def test_epoch_end()

    # def loss_func(): # TODO check loss calculation

    def process_batch(self, batch):
        target_smiles = batch.pop('target_smiles')

        # TODO check correctness of mask assignment, dimensions, and slicing
        batch['encoder_input'] = batch['encoder_input']
        batch['encoder_pad_mask'] = batch['encoder_pad_mask'].long()
        batch['decoder_input'] = batch['decoder_input']
        batch['decoder_pad_mask'] = batch['decoder_pad_mask'].long()

        keys = ['encoder_input', 'decoder_input', 'encoder_pad_mask', 'decoder_pad_mask']
        datatype = torch.int64
        data_b = tensor_parallel.broadcast_data(keys, batch, datatype)

        tokens_enc = data_b['encoder_input'].long()
        tokens_dec = data_b['decoder_input'].long()

        labels = data_b['decoder_input'].long()
        loss_mask = data_b['decoder_pad_mask'].float() # TODO check that loss mask is correct

        enc_mask = data_b['encoder_pad_mask']
        dec_mask = data_b['decoder_pad_mask']

        return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, target_smiles

    # def predict_step():

    # def decode():

    # def complete --> decode
    