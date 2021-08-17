# coding=utf-8

import math
from functools import partial
from typing import Optional, Tuple
from apex.normalization import FusedLayerNorm
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch

from megatron import mpu
try:
    from megatron.module import MegatronModule # v 1.1.5
except:
    from megatron.model.module import MegatronModule

from dataclasses import dataclass
from nemo.collections.chem.data import MoleculeCsvDatasetConfig
from nemo.collections.chem.decoder import DecodeSamplerConfig
from nemo.core.classes.dataset import DatasetConfig
from nemo.core.config.modelPT import OptimConfig, SchedConfig, ModelConfig

from .megatron_bart_enc_dec import ParallelTransformerDecoder, ParallelTransformerEncoder

# Model parameters
DEFAULT_D_MODEL = 256
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FEEDFORWARD = 4 * DEFAULT_D_MODEL
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_DROPOUT = 0.0


@dataclass
class MegatronBARTSchedConfig(SchedConfig):
    name: str = 'CosineAnnealing'
    last_epoch: int = -1
    warmup_steps: Optional[int] = 8000
    min_lr: Optional[float] = 1.0e-5
    max_steps: Optional[int] = 110000 # TODO this is not in original paper
    monitor: Optional[str] = 'loss'
    reduce_on_plateau: Optional[bool] = False


@dataclass
class MegatronBARTOptimConfig(OptimConfig):
    name: str = 'adam'
    lr: float = 1.0 # TODO this is what was in the paper, I believe it is wrong
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    sched: Optional[SchedConfig] = MegatronBARTSchedConfig()


@dataclass
class MegatronBARTConfig(ModelConfig):
    name: str = 'MegatronBART'
    decode_sampler: DecodeSamplerConfig = DecodeSamplerConfig()
    d_model: int = DEFAULT_D_MODEL
    num_layers: int = DEFAULT_NUM_LAYERS
    num_heads: int = DEFAULT_NUM_HEADS
    d_feedforward: int = DEFAULT_D_FEEDFORWARD
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    dropout: float = DEFAULT_DROPOUT
    pretrained: Optional[bool] = False
    checkpoint_file: Optional[str] = None
    train_ds: DatasetConfig = MoleculeCsvDatasetConfig()
    validation_ds: Optional[DatasetConfig] = None
    test_ds: Optional[DatasetConfig] = None
    optim: Optional[OptimConfig] = MegatronBARTOptimConfig()


class MegatronBART(MegatronModule):

    def __init__(
        self,
        decode_sampler,
        pad_token_idx,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_feedforward,
        max_seq_len,
        dropout
        ):

        super().__init__()

        self.sampler = decode_sampler
        self.pad_token_idx = pad_token_idx
        self.val_sampling_alg = 'greedy'
        self.num_beams = 5
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.emb_dropout = nn.Dropout(p=dropout)
        init_method = init.xavier_uniform_

        self.emb = nn.Embedding(vocab_size, d_model)
        self.dropout = dropout
        self.encoder = ParallelTransformerEncoder(
            self.num_layers,
            self.d_model,
            self.num_heads,
            self.dropout,
            bias=True,
            init_method=init_method,
            )
        self.decoder = ParallelTransformerDecoder(
            self.num_layers,
            self.d_model,
            self.num_heads,
            self.dropout,
            bias=True,
            init_method=init_method,
            )
        self.token_fc = mpu.RowParallelLinear(d_model, vocab_size,
                input_is_parallel=False, init_method=init_method,
                skip_bias_add=False)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none',
                ignore_index=pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self._init_params(init_method)
        self.register_buffer('pos_emb', self._positional_embs())

    def forward(self, x):
        """ Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model 
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """

        encoder_input = x['encoder_input']
        decoder_input = x['decoder_input']
        encoder_pad_mask = x['encoder_pad_mask'].transpose(0, 1)
        decoder_pad_mask = x['decoder_pad_mask'].transpose(0, 1)

        encoder_embs = self._construct_input(encoder_input)
        decoder_embs = self._construct_input(decoder_input)

        (seq_len, _, _) = tuple(decoder_embs.size())
        tgt_mask = \
            self._generate_square_subsequent_mask(seq_len).to(decoder_embs.device)

        memory = self.encoder(encoder_embs,
                              src_key_padding_mask=encoder_pad_mask)
        model_output = self.decoder(decoder_embs, memory,
                                    tgt_mask=tgt_mask,
                                    tgt_key_padding_mask=decoder_pad_mask,
                                    memory_key_padding_mask=encoder_pad_mask.clone())

        token_output, _ = self.token_fc(model_output)
        output = {'model_output': model_output,
                  'token_output': token_output}

        return output

    def encode(self, batch):
        """ Construct the memory embedding for an encoder input

        Args:
            batch (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
            })

        Returns:
            encoder memory (Tensor of shape (seq_len, batch_size, d_model))
        """

        encoder_input = batch['encoder_input']
        encoder_pad_mask = batch['encoder_pad_mask'].transpose(0, 1)
        encoder_embs = self._construct_input(encoder_input)
        model_output = self.encoder(encoder_embs,
                                    src_key_padding_mask=encoder_pad_mask)
        return model_output

    def decode(self, batch):
        """ Construct an output from a given decoder input

        Args:
            batch (dict {
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
                "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
                "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)
            })
        """

        decoder_input = batch['decoder_input']
        decoder_pad_mask = batch['decoder_pad_mask'].transpose(0, 1)
        memory_input = batch['memory_input']
        memory_pad_mask = batch['memory_pad_mask'].transpose(0, 1)

        decoder_embs = self._construct_input(decoder_input)

        (seq_len, _, _) = tuple(decoder_embs.size())
        tgt_mask = \
            self._generate_square_subsequent_mask(seq_len).to(decoder_embs.device)

        model_output = self.decoder(decoder_embs, memory_input,
                                    tgt_key_padding_mask=decoder_pad_mask,
                                    memory_key_padding_mask=memory_pad_mask,
                                    tgt_mask=tgt_mask)
        token_output, _  = self.token_fc(model_output)
        token_probs = self.log_softmax(token_output)
        return token_probs

    def validation_step(self, batch, batch_idx=None):
        self.eval()

        model_output = self.forward(batch)
        target_smiles = batch['target_smiles']

        loss = self._calc_loss(batch, model_output)
        token_acc = self._calc_char_acc(batch, model_output)
        perplexity = self._calc_perplexity(batch, model_output)
        (mol_strs, log_lhs) = self.sample_molecules(batch,
                sampling_alg=self.val_sampling_alg)
        metrics = self.sampler.calc_sampling_metrics(mol_strs,
                target_smiles)

        val_outputs = {
            'val_loss': loss.item(),
            'val_token_acc': token_acc,
            'val_perplexity': perplexity,
            'val_molecular_accuracy': metrics['accuracy'],
            'val_invalid_smiles': metrics['invalid'],
            }
        return val_outputs

    def _calc_loss(self, batch_input, model_output):
        """ Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor),
        """

        tokens = batch_input['target']
        pad_mask = batch_input['target_pad_mask']
        token_output = model_output['token_output']
        token_mask_loss = self._calc_mask_loss(token_output, tokens,
                pad_mask)
        return token_mask_loss

    def _calc_mask_loss(
        self,
        token_output,
        target,
        target_mask,
        ):
        """ Calculate the loss for the token prediction task

        Args:
            token_output (Tensor of shape (seq_len, batch_size, vocab_size)): token output from transformer
            target (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokenizer
            target_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens

        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """

        (seq_len, batch_size) = tuple(target.size())
        token_pred = token_output.reshape((seq_len * batch_size,
                -1)).float()
        loss = self.loss_fn(token_pred,
                            target.reshape(-1)).reshape((seq_len,
                batch_size))
        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens
        return loss

    def _calc_perplexity(self, batch_input, model_output):
        target_ids = batch_input['target']
        target_mask = batch_input['target_pad_mask']
        vocab_dist_output = model_output['token_output']
        inv_target_mask = ~(target_mask > 0)
        log_probs = vocab_dist_output.gather(2,
                target_ids.unsqueeze(2)).squeeze(2)
        log_probs = log_probs * inv_target_mask
        log_probs = log_probs.sum(dim=0)
        seq_lengths = inv_target_mask.sum(dim=0)
        exp = -(1 / seq_lengths)
        perp = torch.pow(log_probs.exp(), exp)
        return perp.mean().item()

    @staticmethod
    def _calc_char_acc(batch_input, model_output):
        token_ids = batch_input['target']
        target_mask = batch_input['target_pad_mask']
        token_output = model_output['token_output']
        target_mask = ~(target_mask > 0)
        (_, pred_ids) = torch.max(token_output.float(), dim=2)
        correct_ids = torch.eq(token_ids, pred_ids)
        correct_ids = correct_ids * target_mask
        num_correct = correct_ids.sum().cpu().detach().item()
        total = target_mask.sum().cpu().detach().item()
        accuracy = num_correct / total
        return accuracy

    def sample_molecules(self, batch_input, sampling_alg='greedy'):
        """ Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        enc_input = batch_input['encoder_input']
        enc_mask = batch_input['encoder_pad_mask']

        # Freezing the weights reduces the amount of memory leakage in the transformer
        #model.eval()

        with torch.no_grad():

            encode_input = {'encoder_input': enc_input,
                            'encoder_pad_mask': enc_mask}
            memory = self.encode(encode_input)
            mem_mask = enc_mask.clone()
            (_, batch_size, _) = tuple(memory.size())
            decode_fn = partial(self._decode_fn, memory=memory,
                                mem_pad_mask=mem_mask)
            #self.sampler.device = self.device
            if sampling_alg == 'greedy':
                (mol_strs, log_lhs) = \
                    self.sampler.greedy_decode(decode_fn, batch_size,device=memory.device)
            elif sampling_alg == 'beam':
                (mol_strs, log_lhs) = \
                    self.sampler.beam_decode(decode_fn, batch_size,
                        self.num_beams,device=memory.device)

        # Must remember to unfreeze!
        #model.train()

        return (mol_strs, log_lhs)

    def _decode_fn(
        self,
        token_ids,
        pad_mask,
        memory,
        mem_pad_mask,
        ):
        decode_input = {
            'decoder_input': token_ids,
            'decoder_pad_mask': pad_mask,
            'memory_input': memory,
            'memory_pad_mask': mem_pad_mask,
            }
        model_output = self.decode(decode_input)
        return model_output

    def _construct_input(self, token_ids, sentence_masks=None):
        (seq_len, _) = tuple(token_ids.size())
        token_embs = self.emb(token_ids)

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.d_model)
        positional_embs = self.pos_emb[:seq_len, :
                ].unsqueeze(0).transpose(0, 1)
        embs = token_embs + positional_embs
        embs = self.emb_dropout(embs)
        return embs

    def _positional_embs(self):
        """ Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor([dim / self.d_model for dim in range(0,
                            self.d_model, 2)])
        encs = 10000 ** encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs))
                for pos in range(self.max_seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[:self.d_model]
                for enc in encs]
        encs = torch.stack(encs)
        return encs

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        """ 
        Method copied from Pytorch nn.Transformer.
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Args:
            sz (int): Size of mask to generate

        Returns:
            torch.Tensor: Square autoregressive mask for decode 
        """

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'
                )).masked_fill(mask == 1, float(0.0))
        return mask

    def _init_params(self, method):
        """
        Apply initialisation of learnable weights
        """

        for p in self.parameters():
            if p.dim() > 1:
                method(p)

