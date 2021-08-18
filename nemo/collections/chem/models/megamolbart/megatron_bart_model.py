
import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import tempfile
import re
import time

import numpy as np
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.utils.data as pt_data
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from nemo.collections.common.data import ConcatDataset
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.app_state import AppState
from nemo.core.classes.common import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import ChannelType, LossType, MaskType, NeuralType
from nemo.utils.env_var_parsing import get_envint
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.utils import logging, model_utils

from megatron import get_args, initialize_megatron
from megatron.model.bert_model import bert_attention_mask_func
from megatron.checkpointing import set_checkpoint_version
from megatron.model import get_language_model
from megatron.mpu import (
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_model_parallel_group,
    model_parallel_is_initialized,
    set_pipeline_model_parallel_rank,
    set_pipeline_model_parallel_world_size,
)

from nemo.collections.chem.data import MoleculeCsvDatasetConfig, MoleculeDataset, MoleculeIterableDataset, ConcatIterableDataset, expand_dataset_paths
from nemo.collections.chem.tokenizer import MolEncTokenizer
from nemo.collections.chem.decoder import DecodeSampler
from .megatron_bart_base import MegatronBART
from pysmilesutils.augment import SMILESAugmenter
from copy import copy

__all__ = ["MegaMolBARTModel"]


class MegaMolBARTModel(ModelPT):   
    def __init__(self, cfg: DictConfig, trainer: pl.Trainer = None) -> None:

        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        self._set_app_state(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        self._model_name = cfg.model.name
        self._restore_path = cfg.model.checkpoint_file
        self._hidden_size = cfg.model.d_model
        self.max_seq_len = cfg.model.max_seq_len

        if not os.path.exists(cfg.tokenizer.vocab_path):
            raise ValueError(f'Vocab file not found at {cfg.tokenizer.vocab_path}')
        self.tokenizer = self.setup_tokenizer(cfg.tokenizer)
        self._vocab_size = len(self.tokenizer)

        _ = self.setup_megatron(cfg) # Megatron initialization -- must be done before superclass init and model loaded            
        super().__init__(cfg=cfg.model, trainer=trainer)
        self.config = OmegaConf.create(cfg.model)

        # Load model
        sampler = self.setup_sampler(self.tokenizer, cfg)
        pad_token_idx = self.tokenizer.vocab[self.tokenizer.pad_token]
        self.model = MegatronBART( 
                                sampler,
                                pad_token_idx,
                                self._vocab_size,
                                cfg.model.d_model,
                                cfg.model.num_layers,
                                cfg.model.num_heads,
                                cfg.model.d_feedforward,
                                cfg.model.max_seq_len,
                                cfg.model.dropout)

        self.num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.setup_optimization(cfg.model.optim) # TODO check warning from training
        self.aug = SMILESAugmenter()

    def _set_app_state(self, cfg):
        app_state = AppState()
        if cfg.trainer is not None:
            app_state._world_size = cfg.trainer.num_nodes * cfg.trainer.gpus
            num_gpus = cfg.trainer.gpus
        else:
            app_state._world_size = 1
            num_gpus = 1

        env = os.environ.copy()
        app_state._local_rank = int(env.get('LOCAL_RANK', 0))
        app_state._node_rank = int(env.get('NODE_RANK', 0))
        app_state._global_rank = app_state._local_rank + (app_state._node_rank * num_gpus) # TODO better way to calculate?
        app_state._model_parallel_size = None
        app_state._model_parallel_rank = None
        # app_state._device_id #          TODO add these
        # app_state._model_parallel_group
        # app_state._data_parallel_size
        # app_state._data_parallel_rank
        # app_state._data_parallel_group
        self._app_state = app_state

    @staticmethod
    def _update_megatron_args(
        micro_batch_size: int = 1,
        tensor_model_parallel_size: int = 1,
        scaled_masked_softmax_fusion: bool = False,
        bias_gelu_fusion: bool = False,
        bias_dropout_fusion: bool = False):

        def extra_args_provider(parser):
            parser.set_defaults(micro_batch_size=micro_batch_size)
            parser.set_defaults(tensor_model_parallel_size=tensor_model_parallel_size)
            parser.set_defaults(scaled_masked_softmax_fusion=scaled_masked_softmax_fusion)
            parser.set_defaults(bias_gelu_fusion=bias_gelu_fusion)
            parser.set_defaults(bias_dropout_fusion=bias_dropout_fusion)
            return parser

        return extra_args_provider

    @staticmethod
    def _get_megatron_vocab_file() -> str:
        """Generate fake Megatron vocab file with required tokens"""
        fake_vocab_contents = '\n'.join(['[CLS]', '[SEP]', '[PAD]', '[MASK]'])
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as fh:
            fh.write(fake_vocab_contents)
            vocab_file = fh.name
        return vocab_file

    def complete_lazy_init(self) -> None:
        # Finish megatron-lm initialization
        if hasattr(self, "_lazy_init_fn") and self._lazy_init_fn is not None:
            logging.info('Completing lazy initialization of Megatron framework...')
            self._lazy_init_fn()
            self._lazy_init_fn = None

    def setup_megatron(self, cfg: DictConfig) -> dict:
        """Initialize Megatron"""
        model_parallel_size = self._app_state._model_parallel_size
        model_parallel_rank = self._app_state._model_parallel_rank

        # Configure globals
        set_pipeline_model_parallel_rank(0)  # Pipeline model parallelism not currently implemented in NeMo
        set_pipeline_model_parallel_world_size(1)  # Pipeline model parallelism not currently implemented in NeMo

        # megatron input arguments
        args = {'num_layers': cfg.model.num_layers,
                'hidden_size': cfg.model.d_model,
                'num_attention_heads': cfg.model.num_heads,
                'max_position_embeddings': cfg.model.max_seq_len,
                'onnx_safe': True,
                'lazy_mpu_init': True,
                'tokenizer_type': 'BertWordPieceCase',
                'vocab_file': self._get_megatron_vocab_file()}
                # TODO vocab size may need to be set

        # extra args provider
        if model_parallel_size is not None:
            app_state = AppState()
            self._app_state = app_state
            os.environ["WORLD_SIZE"] = str(app_state.world_size) # Must be set for model parallel megatron-lm
            os.environ["RANK"] = str(model_parallel_rank)
            extra_args_provider = self._update_megatron_args(tensor_model_parallel_size=model_parallel_size)
        else:
            extra_args_provider = self._update_megatron_args()

        # Initialize part of Megatron global state that is needed for its constructor.
        # We set 'lazy_mpu_init' flag on to make Megatron do only the initialization that does not depend
        # on ddp be initialized yet (and we don't want Megatron to initialize DDP itself either)
        # and to return a hook for us to call after PTL has torch.distributed initialized.
        # (or if no PTL in case of inference - then we'll initialize torch.distributed)
        # We call and clear this hook on first call to forward()
        self._lazy_init_fn = initialize_megatron(
            extra_args_provider=extra_args_provider, args_defaults=args, ignore_unknown_args=True
        )

        # Read Megatron arguments back
        args = get_args()
        logging.info(f'Megatron-lm argparse args: {args}')

        # This loads a fake model from megatron, mostly for the sake of ensuring compatible checkpoint dict
        _, self._language_model_key = get_language_model(
            attention_mask_func=bert_attention_mask_func, num_tokentypes=2, add_pooler=False
        )
        return args

    def _load_checkpoint(self, filename: str) -> None:
        """Helper function for loading megatron checkpoints.

        Args:
            filename (str): Path to megatron checkpoint.
        """
        state_dict = torch.load(filename, map_location='cpu')
        if 'checkpoint_version' in state_dict:
            if state_dict['checkpoint_version'] is not None:
                set_checkpoint_version(state_dict['checkpoint_version'])
                logging.info(
                    f"Megatron-lm checkpoint version found. Setting checkpoint_version to {state_dict['checkpoint_version']}."
                )
        else:
            logging.warning('Megatron-lm checkpoint version not found. Setting checkpoint_version to 0.')
            set_checkpoint_version(0)
        # To load from Megatron pretrained checkpoint
        if 'model' in state_dict:
            self.language_model.load_state_dict(state_dict['model'][self._language_model_key])
        else:
            self.load_state_dict(state_dict)

        logging.info(f"Checkpoint loaded from from {filename}")

    def restore_weights(self, restore_path: str) -> None:
        """Restores module/model's weights for Megatron model
           For model parallel checkpoints the directory structure
           should be restore_path/mp_rank_0X/model_optim_rng.pt

        Args:
            restore_path (str): restore_path should a file or a directory if using model parallel
        """
        self._restore_path = restore_path
        if os.path.isfile(restore_path):
            self._load_checkpoint(restore_path)
        elif os.path.isdir(restore_path):
            # Need model parallel groups to restore model parallel checkpoints
            if model_parallel_is_initialized():
                model_parallel_rank = dist.get_rank(group=get_model_parallel_group())
                mp_restore_path = f'{restore_path}/mp_rank_{model_parallel_rank:02d}/model_optim_rng.pt'
                self._load_checkpoint(mp_restore_path)
            else:
                logging.info(f'torch.distributed not initialized yet. Will not restore model parallel checkpoint')
        else:
            logging.error(f'restore_path: {restore_path} must be a file or directory.')

    @staticmethod
    def setup_tokenizer(cfg: DictConfig) -> MolEncTokenizer:
        tokenizer = MolEncTokenizer.from_vocab_file(**cfg)
        return tokenizer

    @staticmethod
    def setup_sampler(tokenizer: MolEncTokenizer, cfg: DictConfig) -> DecodeSampler:
        return DecodeSampler(tokenizer, cfg.model.max_seq_len)

    def setup_training_data(self, train_data_config: Optional[DictConfig]) -> None:
        logging.info('Loading training data')
        self._train_dl = self.setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if val_data_config.get('filepath'): # TODO skip if limit_val_batches=0.0
            logging.info('Loading validation data')
            self._validation_dl = self.setup_dataloader_from_config(cfg=val_data_config)
            # Instantiate Torchmetric for each val dataloader
            if self._validation_dl is not None:
                for dataloader_idx in range(len(self._validation_dl)):
                    if dataloader_idx == 0:
                        setattr(
                            self, f'val_loss', GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True),
                        )
                    else:
                        setattr(
                            self,
                            f'val_loss_{dataloader_idx}',
                            GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True),
                        )
        else:
            logging.info('Skipping validation data loading')

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        logging.info('Loading test data')
        self._test_dl = self.setup_dataloader_from_config(cfg=test_data_config)
        # instantiate Torchmetric for each test dataloader
        if self._test_dl is not None:
            for dataloader_idx in range(len(self._test_dl)):
                if dataloader_idx == 0:
                    setattr(
                        self, f'test_loss', GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True),
                    )
                else:
                    setattr(
                        self,
                        f'test_loss_{dataloader_idx}',
                        GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True),
                    )

    def _setup_dataset_from_config(self, cfg: DictConfig):
        cfg = dict(cfg.copy())
        filepath = cfg.pop('filepath', None)
        use_iterable = cfg.pop('use_iterable', False)

        dataset_paths = expand_dataset_paths(filepath)
        logging.info(f'Loading data from {dataset_paths}')
        datasets = []
        for path in dataset_paths:
            if use_iterable:
                data = MoleculeIterableDataset(filepath=path, **cfg)
            else:
                data = MoleculeDataset(filepath=path, **cfg)
            datasets.append(data)

        if len(datasets) == 1:
            datasets = datasets[0]
        else:
            if use_iterable:
                datasets = ConcatIterableDataset(datasets)
            else:
                datasets = pt_data.ConcatDataset(datasets)
        return datasets

    def setup_dataloader_from_config(self, cfg: DictConfig):
        use_iterable = cfg.get('use_iterable', False)
        dataset = self._setup_dataset_from_config(cfg)

        if use_iterable:
            sampler = None
        else:
            sampler_name = pt_data.RandomSampler if cfg.shuffle else pt_data.SequentialSampler
            sampler = sampler_name(dataset)
        
        dataloader = pt_data.DataLoader(dataset,
            sampler=sampler,
            batch_size=cfg.batch_size,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False), 
            drop_last=cfg.get("drop_last", False),
            collate_fn=self.collate_fn)

        return dataloader

    @staticmethod
    def _check_seq_len(tokens: List[List[str]], mask: List[List[int]], max_seq_len: int):
        """ Warn user and shorten sequence if the tokens are too long, otherwise return original

        Args:
            tokens (List[List[str]]): List of token sequences
            mask (List[List[int]]): List of mask sequences

        Returns:
            tokens (List[List[str]]): List of token sequences (shortened, if necessary)
            mask (List[List[int]]): List of mask sequences (shortened, if necessary)
        """

        seq_len = max([len(ts) for ts in tokens])
        if seq_len > max_seq_len:
            tokens_short = [ts[:max_seq_len] for ts in tokens]
            mask_short = [ms[:max_seq_len] for ms in mask]
            return (tokens_short, mask_short)
        return (tokens, mask)

    def augmenter(self, mol): # TODO move to dataset
        try:
            aug_smi = self.aug(mol)[0]
        except:
            aug_smi = mol
        return aug_smi

    def collate_fn(self, batch):
        """ Used by DataLoader to concatenate/collate inputs."""
        encoder_smiles = copy(batch)
        decoder_smiles = batch

        encoder_smiles = [self.augmenter(x) for x in encoder_smiles]
        decoder_smiles = [self.augmenter(x) for x in decoder_smiles]

        enc_token_output = self.tokenizer.tokenize(encoder_smiles, mask=True, pad=True)
        dec_token_output = self.tokenizer.tokenize(decoder_smiles, pad=True)

        enc_mask = enc_token_output['masked_pad_masks']
        enc_tokens = enc_token_output['masked_tokens']
        dec_tokens = dec_token_output['original_tokens']
        dec_mask = dec_token_output['original_pad_masks']

        (enc_tokens, enc_mask) = self._check_seq_len(enc_tokens, enc_mask, self.max_seq_len)
        (dec_tokens, dec_mask) = self._check_seq_len(dec_tokens, dec_mask, self.max_seq_len)

        enc_token_ids = self.tokenizer.convert_tokens_to_ids(enc_tokens)
        dec_token_ids = self.tokenizer.convert_tokens_to_ids(dec_tokens)

        enc_token_ids = torch.tensor(enc_token_ids).transpose(0, 1)
        enc_pad_mask = torch.tensor(enc_mask,
                                    dtype=torch.int64).transpose(0, 1)
        dec_token_ids = torch.tensor(dec_token_ids).transpose(0, 1)
        dec_pad_mask = torch.tensor(dec_mask,
                                    dtype=torch.int64).transpose(0, 1)

        collate_output = {
            'encoder_input': enc_token_ids,
            'encoder_pad_mask': enc_pad_mask,
            'decoder_input': dec_token_ids[:-1, :],
            'decoder_pad_mask': dec_pad_mask[:-1, :],
            'target': dec_token_ids.clone()[1:, :],
            'target_pad_mask': dec_pad_mask.clone()[1:, :],
            'target_smiles': decoder_smiles,
            }

        return collate_output

    @typecheck()
    def forward(self, batch):
        app_state = AppState()
        if app_state.model_parallel_size is None:
            self.complete_lazy_init()

        outputs = self.model(batch)
        return outputs

    def training_step(self, batch: dict, batch_idx: int) -> Dict:
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`. 
        """
        start_time = time.monotonic()

        outputs = self.forward(batch)
        loss = self.model._calc_loss(batch, outputs)
        char_acc = self.model._calc_char_acc(batch, outputs)
        lr = self._optimizer.param_groups[0]["lr"]

        end_time = time.monotonic()
        duration = end_time - start_time


        self.log('lr', lr)
        self.log('train_loss', loss.item(), on_epoch=True)
        self.log('train_char_acc', char_acc, on_epoch=True, sync_dist=True)
        self.log('step_time', duration, on_step=True)

        tensorboard_logs = {'train_loss': loss.item(),
                            'train_char_acc': char_acc, 
                            'lr': lr,
                            'step_time': duration}

        return {'loss': loss, 
                'log': tensorboard_logs}

    @rank_zero_only
    def log_param_stats(self):
        for name, p in self.named_parameters():
            if p.requires_grad:
                self.trainer.logger.experiment.add_histogram(name + '_hist', p, global_step=self.global_step)
                self.trainer.logger.experiment.add_scalars(
                    name,
                    {'mean': p.mean(), 'stddev': p.std(), 'max': p.max(), 'min': p.min()},
                    global_step=self.global_step,
                )

    def validation_step(self, batch: dict, batch_idx: int) -> Dict:
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`. 
        """
        outputs = self.model.validation_step(batch)
        loss = outputs['val_loss']
        char_acc = outputs['val_token_acc']
        molecular_accuracy = outputs['val_molecular_accuracy']
        perplexity = outputs['val_perplexity']
        invalid_smiles = outputs['val_invalid_smiles']

        tensorboard_logs = {'val_loss': loss,
                            'val_char_acc': char_acc,
                            'val_mol_acc': molecular_accuracy,
                            'val_perplexity': perplexity,
                            'val_invalid_smiles': invalid_smiles}
        self.log_dict(tensorboard_logs, on_epoch=True, sync_dist=True)
        return {'val_loss': loss, 
                'char_acc': char_acc,
                'mol_acc': molecular_accuracy,
                'perplexity': perplexity,
                'invalid_smiles': invalid_smiles,
                'log': tensorboard_logs}

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        loss = torch.tensor([x['val_loss'] for x in outputs]).mean().item()
        perplexity = torch.tensor([x['perplexity'] for x in outputs]).mean().item()
        tensorboard_logs = {'val_loss': loss, 'perplexity': perplexity}
        logging.info(f'Validation perplexity {perplexity}')
        return {'val_loss': loss, 'log': tensorboard_logs}

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
