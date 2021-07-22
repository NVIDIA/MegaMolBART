# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2020, MeetKai Inc.  All rights reserved.
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

import sys
sys.path.insert(0, '/workspace') # TODO remove this
sys.path.insert(0, '/code/NeMo') # TODO remove this

import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import tempfile
import re
import braceexpand

import numpy as np
from omegaconf import DictConfig, OmegaConf

import torch
import torch.distributed as dist
import torch.utils.data as pt_data
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.app_state import AppState
from nemo.core.classes.common import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import ChannelType, LossType, MaskType, NeuralType
from nemo.utils.env_var_parsing import get_envint
from nemo.collections.common.metrics import GlobalAverageLossMetric

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
# from megatron.data.samplers import DistributedBatchSampler
# from torch.utils.data.distributed import DistributedSampler
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin

# TODO simplify imports
from nemo.collections.chem.tokenizer.tokenizer import MolEncTokenizer
from nemo.collections.chem.data.csv_data import MoleculeDataset
from nemo.collections.chem.decoder.decoder import DecodeSampler
from nemo.collections.chem.parts.util import REGEX, DEFAULT_VOCAB_PATH, DEFAULT_CHEM_TOKEN_START #,DEFAULT_MAX_SEQ_LEN #, DEFAULT_VOCAB_PATH, DEFAULT_CHEM_TOKEN_START, REGEX
from nemo.collections.chem.models.megatron_bart import MegatronBART

__all__ = ["MegaMolBARTModel"]

from IPython import embed
class MegaMolBARTModel(ModelPT):
    # @property
    # def input_types(self) -> Optional[Dict[str, NeuralType]]:
    #     return {
    #         "input_ids": NeuralType(('B', 'T'), ChannelType()),
    #         "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
    #         "decoder_input_ids": NeuralType(('B', 'T'), ChannelType(), optional=True),
    #         "labels": NeuralType(('B', 'T'), ChannelType(), optional=True),
    #     }

    # @property
    # def output_types(self) -> Optional[Dict[str, NeuralType]]:
    #     return {
    #         "loss": NeuralType((), LossType()),
    #         "decoder_hidden_states": NeuralType(("B", "T"), ChannelType(), optional=True),
    #         "encoder_hidden_states": NeuralType(("B", "T"), ChannelType(), optional=True),
    #     }

    def __init__(self, cfg: DictConfig, trainer: pl.Trainer = None) -> None:
        
        self._app_state = None
        self._model_name = cfg.model.name
        self._restore_path = cfg.model.checkpoint_file
        self._hidden_size = cfg.model.hidden_size
        self.max_seq_len = cfg.model.max_seq_len

        self._model_parallel_size = None # TODO how to configure -- Megatron set requires them for initialization
        self._model_parallel_rank = None #      which must be done before torch.distributed is intialized
        self.world_size = cfg.trainer.num_nodes * cfg.trainer.gpus if trainer is not None else 1

        if not os.path.exists(cfg.model.tokenizer.vocab_file):
            raise ValueError(f'Vocab file not found at {cfg.model.tokenizer.vocab_file}')
        self.tokenizer = self.setup_tokenizer(cfg.model.tokenizer)
        self._vocab_size = len(self.tokenizer)

        # Megatron initilization -- must be done before superclass init and model loaded
        _ = self.setup_megatron(cfg)
        super().__init__(cfg=cfg.model, trainer=trainer)
        self.config = OmegaConf.create(cfg.model) # TODO verify that checkpoint saving/loading works

        # load model
        sampler = self.setup_sampler(self.tokenizer, cfg)
        pad_token_idx = self.tokenizer.vocab[self.tokenizer.pad_token]
        self.model = MegatronBART( 
                                sampler,
                                pad_token_idx,
                                self._vocab_size,
                                cfg.model.hidden_size,
                                cfg.model.num_layers,
                                cfg.model.num_attention_heads,
                                cfg.model.feedforward,
                                cfg.model.max_seq_len,
                                cfg.model.dropout)

        self.num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.setup_optimization(cfg.optim)

    def _update_megatron_args(
        self,
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

    def _get_megatron_vocab_file(self) -> str:
        """Generate fake Megatron vocab file with required tokens"""
        fake_vocab_contents = '\n'.join(['[CLS]', '[SEP]', '[PAD]', '[MASK]'])
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as fh:
            fh.write(fake_vocab_contents)
            vocab_file = fh.name
        return vocab_file

    def complete_lazy_init(self) -> None:
        # finish megatron-lm initialization
        if hasattr(self, "_lazy_init_fn") and self._lazy_init_fn is not None:
            self._lazy_init_fn()
            self._lazy_init_fn = None

    def setup_megatron(self, cfg: DictConfig) -> dict:
        """Initialize Megatron"""
        # configure globals
        set_pipeline_model_parallel_rank(0)  # pipeline model parallelism not implemented in NeMo
        set_pipeline_model_parallel_world_size(1)  # pipeline model parallelism not implemented in NeMo

        # megatron input arguments
        args = {'num_layers': cfg.model.num_layers,
                'hidden_size': cfg.model.hidden_size,
                'num_attention_heads': cfg.model.num_attention_heads,
                'max_position_embeddings': cfg.model.max_seq_len,
                'onnx_safe': True,
                'lazy_mpu_init': True,
                'tokenizer_type': 'BertWordPieceCase',
                'vocab_file': self._get_megatron_vocab_file()}
                # TODO vocab size may need to be set

        # extra args provider
        if self._model_parallel_size is not None:
            app_state = AppState()
            self._app_state = app_state
            # must be set for model parallel megatron-lm
            os.environ["WORLD_SIZE"] = str(app_state.world_size)
            os.environ["RANK"] = str(self._model_parallel_rank)
            extra_args_provider = self._update_megatron_args(tensor_model_parallel_size=self._model_parallel_size)
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

        # read Megatron arguments back
        args = get_args()
        logging.info(f'Megatron-lm argparse args: {args}')

        # This loads a fake model from megatron, mostly for the sake of ensuring compatile checkpoint dict
        # TODO test if needed
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
        # to load from Megatron pretrained checkpoint
        if 'model' in state_dict:
            self.language_model.load_state_dict(state_dict['model'][self._language_model_key])
        else:
            self.load_state_dict(state_dict)

        logging.info(f"Checkpoint loaded from from {filename}")

    def restore_weights(self, restore_path: str) -> None:
        """Restores module/model's weights.
           For model parallel checkpoints the directory structure
           should be restore_path/mp_rank_0X/model_optim_rng.pt

        Args:
            restore_path (str): restore_path should a file or a directory if using model parallel
        """
        self._restore_path = restore_path

        if os.path.isfile(restore_path):
            self._load_checkpoint(restore_path)
        elif os.path.isdir(restore_path):
            # need model parallel groups to restore model parallel checkpoints
            if model_parallel_is_initialized():
                model_parallel_rank = torch.distributed.get_rank(group=get_model_parallel_group())
                mp_restore_path = f'{restore_path}/mp_rank_{model_parallel_rank:02d}/model_optim_rng.pt'
                self._load_checkpoint(mp_restore_path)
            else:
                logging.info(f'torch.distributed not initialized yet. Will not restore model parallel checkpoint')
        else:
            logging.error(f'restore_path: {restore_path} must be a file or directory.')

    def setup_tokenizer(self, cfg: DictConfig) -> MolEncTokenizer:
        regex = cfg.get('regex', REGEX)
        chem_token_start = cfg.get('chem_token_start', DEFAULT_CHEM_TOKEN_START)
        tokenizer = MolEncTokenizer.from_vocab_file(vocab_path=cfg.vocab_file, 
                                                    regex=regex, 
                                                    chem_tokens_start_idx=chem_token_start)
        return tokenizer

    def setup_sampler(self, tokenizer: MolEncTokenizer, cfg: DictConfig) -> DecodeSampler:
        return DecodeSampler(tokenizer, cfg.model.max_seq_len)

    def setup_training_data(self, train_data_config: Optional[DictConfig]) -> None:
        logging.info('Loading training data')
        self._train_dl = self.setup_dataloader_from_config(cfg=train_data_config, split_name='train')

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        logging.info('Loading validation data')
        self._validation_dl = self.setup_dataloader_from_config(cfg=val_data_config, split_name='val')
        # instantiate Torchmetric for each val dataloader
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

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        logging.info('Loading test data')
        self._test_dl = self.setup_dataloader_from_config(cfg=test_data_config, split_name='test')
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

    def _split_dataset_paths_for_ddp(self, filepath):
        # Get all file paths
        if os.path.isdir(filepath):
            dataset_paths = [os.path.join(filepath, x) for x in sorted(os.listdir(filepath))]
        else:
            filepath = re.sub(r"""\(|\[|\<|_OP_""", '{', filepath) # Replace '(', '[', '<' and '_OP_' with '{'
            filepath = re.sub(r"""\)|\]|\>|_CL_""", '}', filepath) # Replace ')', ']', '>' and '_CL_' with '}'
            dataset_paths = list(braceexpand.braceexpand(filepath))
        dataset_paths = np.array(dataset_paths)
        num_dataset_paths = len(dataset_paths)

        # split for data parallel
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size > num_dataset_paths:
                logging.warning(f'World size ({world_size}) is larger than number of data files ({num_dataset_paths}). Data will be duplicated.')
            rank = torch.distributed.get_rank()
        else:
            world_size = 1
            rank = 0
        num_chunks = min(num_dataset_paths, world_size)
        split_dataset_paths = np.array_split(dataset_paths, num_chunks)
        index = rank % len(split_dataset_paths)
        dataset_paths = split_dataset_paths[index]
            
        return dataset_paths

    def _setup_dataset_from_config(self, cfg: DictConfig, split_name: str):
        dataset_paths = self._split_dataset_paths_for_ddp(cfg.filepath)
        logging.info(f'Loading data from {dataset_paths}')

        datasets = []
        for path in dataset_paths:
            data = MoleculeDataset(path, split=split_name, zinc=cfg.zinc)
            datasets.append(data)

        if len(datasets) == 1:
            datasets = datasets[0]
        else:
            datasets = torch.utils.data.ConcatDataset(datasets)

        return datasets

    def setup_dataloader_from_config(self, cfg: DictConfig, split_name: str):
        dataset = self._setup_dataset_from_config(cfg, split_name)

        # TODO setup distributed sampler when data loding is improved
        if cfg.shuffle:
            sampler = pt_data.RandomSampler(dataset)
        else:
            sampler = pt_data.SequentialSampler(dataset)

        dataloader = pt_data.DataLoader(dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.get("num_workers", 1),
            pin_memory=cfg.get("pin_memory", False), 
            drop_last=cfg.get("drop_last", False),
            collate_fn=self.collate_fn)

        return dataloader

    def _check_seq_len(self, tokens: List[List[str]], mask: List[List[int]], max_seq_len: int):
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

    def collate_fn(self, batch):
        """ Used by DataLoader to concatenate/collate inputs."""

        encoder_smiles = [x['encoder_smiles'][0] for x in batch]
        decoder_smiles = [x['decoder_smiles'][0] for x in batch]
        enc_token_output = self.tokenizer.tokenize(encoder_smiles, mask=True, pad=True)
        dec_token_output = self.tokenizer.tokenize(decoder_smiles, pad=True)

        enc_mask = enc_token_output['masked_pad_masks']
        enc_tokens = enc_token_output['masked_tokens']
        dec_tokens = dec_token_output['original_tokens']
        dec_mask = dec_token_output['original_pad_masks']

        max_seq_len = self.max_seq_len
        (enc_tokens, enc_mask) = self._check_seq_len(enc_tokens, enc_mask, max_seq_len)
        (dec_tokens, dec_mask) = self._check_seq_len(dec_tokens, dec_mask, max_seq_len)

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

        # batch = get_batch(data_iterator)
        outputs = self.model(batch)
        return outputs

    def training_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`. 
        """
        outputs = self.forward(batch)

        loss = self.model._calc_loss(batch, outputs)
        char_acc = self.model._calc_char_acc(batch, outputs)
        lr = self._optimizer.param_groups[0]["lr"]
        tensorboard_logs = {'train_loss': loss.item(),
                            'val_loss': loss.item(), 
                            'train_char_acc': char_acc, 
                            'lr': lr}

        return {'loss': loss, 
                'log': tensorboard_logs}

    # TODO is this needed for ddp?
    # @rank_zero_only
    # def log_param_stats(self):
    #     for name, p in self.named_parameters():
    #         if p.requires_grad:
    #             self.trainer.logger.experiment.add_histogram(name + '_hist', p, global_step=self.global_step)
    #             self.trainer.logger.experiment.add_scalars(
    #                 name,
    #                 {'mean': p.mean(), 'stddev': p.std(), 'max': p.max(), 'min': p.min()},
    #                 global_step=self.global_step,
    #             )

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`. 
        """
        outputs = self.model.validation_step(batch)
        loss = outputs['val_loss']
        char_acc = outputs['val_token_acc']
        molecular_accuracy= outputs['val_molecular_accuracy']
        perplexity= outputs['val_perplexity']
        invalid_smiles = outputs['val_invalid_smiles']

        tensorboard_logs = {'val_loss': loss,
                            'val_char_acc': char_acc,
                            'val_mol_acc': molecular_accuracy,
                            'val_perplexity': perplexity,
                            'val_invalid_smiles': invalid_smiles}
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
        # TODO is this needed for ddp?
        # # if user specifies one validation dataloader, then PTL reverts to giving a list of dictionary instead of a list of list of dictionary
        # if isinstance(outputs[0], dict):
        #     outputs = [outputs]

        # loss_list, perplexity_list = [], []
        # for dataloader_idx, output in enumerate(outputs):
        #             if dataloader_idx == 0:
        #                 loss = getattr(self, 'val_loss').compute()
        #             else:
        #                 loss = getattr(self, f'val_loss_{dataloader_idx}').compute()

        loss = torch.tensor([x['val_loss'] for x in outputs]).mean().item()
        perplexity = torch.tensor([x['perplexity'] for x in outputs]).mean().item()
        tensorboard_logs = {'val_loss': loss, 'perplexity': perplexity}
        logging.info(f'Validation perplexity {perplexity}')
        return {'val_loss': loss, 'log': tensorboard_logs}

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass


if __name__ == '__main__':
    seed = 42
    pl.seed_everything(seed, workers=True)
    
    cfg = OmegaConf.load('/code/NeMo/examples/chem/conf/megamolbart_base.yaml')
    logging.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    

    nlpddp = NLPDDPPlugin(num_nodes=cfg.trainer.num_nodes)
    
    # nlpddp.lightning_module.has_megatron_encoder = True
    # TODO the above is a failed attempt at working around the following error
    # originating from the NLPDDPPlugin:
    # Traceback (most recent call last):
    # File "nemo/collections/chem/models/megatron_bart_nemo.py", line 540, in <module>
    #     model = MegaMolBARTModel(cfg, trainer)
    # File "/opt/conda/lib/python3.8/site-packages/pytorch_lightning/trainer/trainer.py", line 460, in fit
    #     self._run(model)
    # File "/opt/conda/lib/python3.8/site-packages/pytorch_lightning/trainer/trainer.py", line 758, in _run
    #     self.dispatch()
    # File "/opt/conda/lib/python3.8/site-packages/pytorch_lightning/trainer/trainer.py", line 799, in dispatch
    #     self.accelerator.start_training(self)
    # File "/opt/conda/lib/python3.8/site-packages/pytorch_lightning/accelerators/accelerator.py", line 96, in start_training
    #     self.training_type_plugin.start_training(trainer)
    # File "/code/NeMo/nemo/collections/nlp/parts/nlp_overrides.py", line 67, in start_training
    #     if self.lightning_module.has_megatron_encoder:
    # File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1130, in __getattr__
    #     raise AttributeError("'{}' object has no attribute '{}'".format(
    # AttributeError: 'MegaMolBARTModel' object has no attribute 'has_megatron_encoder'
    
    trainer = pl.Trainer(plugins=[nlpddp],
                         **cfg.trainer, 
                         limit_train_batches=10, 
                         limit_val_batches=2, 
                         limit_test_batches=2)
    

    exp_manager(trainer, cfg.get("exp_manager", None))
    model = MegaMolBARTModel(cfg, trainer)
    trainer.fit(model)


    # @typecheck.disable_checks()
    # def test_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
    #     """Lightning calls this inside the test loop with data from the test dataloader."""
    #     input_ids, input_mask, decoder_input_ids, labels = batch
    #     sequences = self.generate(input_ids=input_ids)
    #     return sequences

    # @typecheck.disable_checks()
    # def test_epoch_end(self, outputs: List[torch.Tensor]) -> Dict[str, List[str]]:
    #     """Called at the end of test to aggregate outputs and decode them."""
    #     texts = [self.encoder_tokenizer.ids_to_text(seq) for batch in outputs for seq in batch]
    #     return {"texts": texts}