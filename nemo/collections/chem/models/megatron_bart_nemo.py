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

import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from transformers import AutoModel, BartForConditionalGeneration, EncoderDecoderModel

from nemo.collections.common.metrics import Perplexity
from nemo.collections.nlp.data.neural_machine_translation import NeuralMachineTranslationDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes.common import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import ChannelType, LossType, MaskType, NeuralType
from nemo.utils import logging
from nemo.collections.nlp.modules.common.megatron import MegatronBertEncoder

from megatron import get_args, initialize_megatron
from megatron.checkpointing import set_checkpoint_version
from megatron.mpu import (
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_model_parallel_group,
    model_parallel_is_initialized,
    set_pipeline_model_parallel_rank,
    set_pipeline_model_parallel_world_size,
)
# from megatron.data.samplers import DistributedBatchSampler
from torch.utils.data.distributed import DistributedSampler

from tokenizer import MolEncTokenizer
from csv_data import MoleculeDataset
from util import REGEX, DEFAULT_VOCAB_PATH, DEFAULT_CHEM_TOKEN_START #,DEFAULT_MAX_SEQ_LEN #, DEFAULT_VOCAB_PATH, DEFAULT_CHEM_TOKEN_START, REGEX

__all__ = ["MegaMolBARTModel"]


class MegatronModel(MegatronBertEncoder):
    def __init__(self, cfg: DictConfig, **kwargs):
        super().__init__(model_name='MegaMolBART', 
                         config=cfg, 
                         vocab_file=cfg.tokenizer.vocab_file, 
                         model_parallel_size=None, 
                         model_parallel_rank=None)
    def forward(self):
        raise AttributeError("Not implemented")


class MegaMolBARTModel(ModelPT, MegatronModel):
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
    #         "decoder_hidden_states": NeuralType(("B", "T", "D"), ChannelType(), optional=True),
    #         "encoder_hidden_states": NeuralType(("B", "T", "D"), ChannelType(), optional=True),
    #     }

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg.model, trainer=trainer)

        self.tokenizer = self.setup_tokenizer(cfg.model.tokenizer)

        # # must assign modules after init
        # if cfg.language_model.pretrained_model_name:
        #     # Setup end-to-end model
        #     if "bart" in cfg.language_model.pretrained_model_name:
        #         self.model = BartForConditionalGeneration.from_pretrained(cfg.language_model.pretrained_model_name)
        #     else:
        #         self.model = AutoModel.from_pretrained(cfg.language_model.pretrained_model_name)
        # else:
        #     if not (
        #         cfg.language_model.pretrained_encoder_model_name and cfg.language_model.pretrained_decoder_model_name
        #     ):
        #         raise ValueError("Both encoder and decoder must be specified")

        #     # Setup encoder/decoder model
        #     self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        #         encoder=cfg.language_model.pretrained_encoder_model_name,
        #         decoder=cfg.language_model.pretrained_decoder_model_name,
        #     )

        # self.validation_perplexity = Perplexity(compute_on_step=False)

        # self.setup_optimization(cfg.optim)

    # @typecheck()
    # def forward(
    #     self,
    #     input_ids: torch.Tensor,
    #     attention_mask: torch.Tensor = None,
    #     decoder_input_ids: torch.Tensor = None,
    #     labels: torch.Tensor = None,
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    #     """
    #     No special modification required for Lightning, define it as you normally would
    #     in the `nn.Module` in vanilla PyTorch.
    #     """
    #     outputs = self.model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         decoder_input_ids=decoder_input_ids,
    #         labels=labels,
    #         return_dict=False,
    #     )
    #     return outputs

    # @typecheck.disable_checks()
    # def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
    #     """Wraps huggingface EncoderDecoder.generate()."""
    #     outputs = self.model.generate(
    #         input_ids=input_ids,
    #         pad_token_id=self.encoder_tokenizer.pad_id,
    #         bos_token_id=self.encoder_tokenizer.bos_id,
    #         eos_token_id=self.encoder_tokenizer.eos_id,
    #         decoder_start_token_id=self.decoder_tokenizer.bos_id,
    #         **self._cfg.generate,
    #     )
    #     return outputs

    # def training_step(self, batch: Tuple, batch_idx: int) -> Dict:
    #     """
    #     Lightning calls this inside the training loop with the data from the training dataloader
    #     passed in as `batch`. Loss calculation from HuggingFace's BartForConditionalGeneration.
    #     """
    #     input_ids, input_mask, decoder_input_ids, labels = batch
    #     loss = self.forward(
    #         input_ids=input_ids, attention_mask=input_mask, decoder_input_ids=decoder_input_ids, labels=labels,
    #     )[0]

    #     tensorboard_logs = {"train_loss": loss, "lr": self._optimizer.param_groups[0]["lr"]}

    #     return {"loss": loss, "log": tensorboard_logs}

    # def validation_step(self, batch: Tuple, batch_idx: int) -> Dict:
    #     """
    #     Lightning calls this inside the validation loop with the data from the validation dataloader
    #     passed in as `batch`. Loss calculation from HuggingFace's BartForConditionalGeneration.
    #     """
    #     input_ids, input_mask, decoder_input_ids, labels = batch
    #     loss, logits = self.forward(
    #         input_ids=input_ids, attention_mask=input_mask, decoder_input_ids=decoder_input_ids, labels=labels,
    #     )[:2]

    #     self.validation_perplexity(logits=logits)

    #     tensorboard_logs = {"val_loss": loss}

    #     return {"val_loss": loss, "log": tensorboard_logs}

    # def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
    #     """
    #     Called at the end of validation to aggregate outputs.
    #     :param outputs: list of individual outputs of each validation step.
    #     """
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     perplexity = self.validation_perplexity.compute()
    #     tensorboard_logs = {"val_loss": avg_loss, "perplexity": perplexity}
    #     logging.info(f"evaluation perplexity {perplexity.item()}")
    #     return {"val_loss": avg_loss, "log": tensorboard_logs}

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

    def setup_tokenizer(self, cfg: DictConfig):
        vocab_file = Path(cfg.get('vocab_file', DEFAULT_VOCAB_PATH))
        regex = cfg.get('regex', REGEX)
        chem_token_start = cfg.get('chem_token_start', DEFAULT_CHEM_TOKEN_START)
        tokenizer = MolEncTokenizer.from_vocab_file(vocab_path=vocab_file, regex=regex, chem_tokens_start_idx=chem_token_start)
        return tokenizer

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self.setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self.setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self.setup_dataloader_from_config(cfg=test_data_config)

    def get_data(self):
        return (self._train_dl, self._validation_dl)

    def _read_dir_df(self, path):
        args = get_args()
        names = os.listdir(path)
        m = len(names)
        world_size = max(get_data_parallel_world_size(), args.world_size)
        rank = max(get_data_parallel_rank(), args.rank)
        partition = int(m/world_size) + 1
        partition = max(partition, 10)
        idx = partition * rank % m
        selected_names = names[idx:(idx+10)]
        dfs = [pd.read_csv(os.path.join(path, f)) for f in selected_names]

        zinc_df = pd.concat(dfs, ignore_index=True, copy=False)
        return zinc_df

    def _setup_dataset_from_config(self, cfg: DictConfig):
        filepath = Path(cfg.filepath) 
        if filepath.is_dir(): # TODO this should be moved into the read_many NEMO class
            df = self._read_dir_df(filepath)
        else:
            df = pd.read_csv(filepath)

        return MoleculeDataset(df, split=cfg.split_name, zinc=cfg.zinc)

    def setup_dataloader_from_config(self, cfg: DictConfig):
        dataset = self._setup_dataset_from_config(cfg)

        if cfg.split_name == 'train':
            
            
            world_size = torch.distributed.get_world_size(group=get_data_parallel_group())
            rank = torch.distributed.get_rank(group=get_data_parallel_group())
            # sampler = torch.utils.data.SequentialSampler(dataset)
            #batch_sampler = DistributedBatchSampler(sampler, cfg.batch_size, True, rank, world_size)
            batch_sampler = DistributedSampler(dataset, rank=rank, shuffle=cfg.shuffle, drop_last=cfg.drop_last)
            
            dataloader = torch.utils.data.DataLoader(dataset,
                batch_sampler=batch_sampler, num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory, collate_fn=self.collate_fn)
        else:
            dataloader = torch.utils.data.DataLoader(dataset,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory, collate_fn=self.collate_fn)

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

        (enc_tokens, enc_mask) = self._check_seq_len(enc_tokens, enc_mask)
        (dec_tokens, dec_mask) = self._check_seq_len(dec_tokens, dec_mask)

        enc_token_ids = self.convert_tokens_to_ids(enc_tokens)
        dec_token_ids = self.convert_tokens_to_ids(dec_tokens)
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

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass


if __name__ == '__main__':
    cfg = OmegaConf.load('/workspace/conf/megamolbart_base.yaml')

    # def extra_args_provider(parser):
    #     parser.set_defaults(micro_batch_size=cfg.model.train_ds.batch_size)
    #     parser.set_defaults(data_parallel_size=1)
    #     parser.set_defaults(num_layers=4)
    #     parser.set_defaults(hidden_size=128)
    #     parser.set_defaults(num_attention_heads=8)
    #     parser.set_defaults(max_position_embeddings=cfg.model.train_ds.max_seq_length)
    #     return parser

    # defaults = {'micro_batch_size': cfg.model.train_ds.batch_size, 
    #         'data_parallel_size': 1,
    #         'num_layers': 4, 
    #         'hidden_size': 128, 
    #         'num_attention_heads': 8,
    #         'max_position_embeddings': cfg.model.train_ds.max_seq_length}

    # initialize_megatron(extra_args_provider=extra_args_provider, ignore_unknown_args=True) #args_defaults=config, 

    # a = MegaMolBARTModel(cfg)
    a = MegatronBertEncoder(cfg)
    from IPython import embed
    embed()