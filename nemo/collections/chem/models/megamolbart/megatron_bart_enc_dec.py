from typing import Optional, Tuple
from apex.normalization import FusedLayerNorm

import torch.nn as nn
from torch.nn import init

from megatron import mpu
try:
    from megatron.module import MegatronModule # v 1.1.5
except:
    from megatron.model.module import MegatronModule

from .megatron_bart_layers import MultiheadAttention, EncoderLayer, DecoderLayer


class ParallelTransformerEncoder(MegatronModule):

    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        init_method=init.xavier_uniform_,
        ):

        super(ParallelTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = dropout
        self.bias = bias
        self.init_method = init_method
        self.layers.extend([self.build_encoder_layer() for i in
                           range(self.num_layers)])
        self.norm = FusedLayerNorm(self.embed_dim)

    def build_encoder_layer(self):
        layer = EncoderLayer(self.embed_dim, self.num_heads,
                             dropout=self.attn_dropout, bias=self.bias,
                             init_method=self.init_method)
        return layer

    def forward(
        self,
        src,
        mask=None,
        src_key_padding_mask=None,
        ):
        """Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Returns:
            encoded output of shape (src_len, batch, embed_dim)
        """

        output = src
        for mod in self.layers:
            output = mod(output, attn_mask=mask,
                         encoder_padding_mask=src_key_padding_mask)
        output = self.norm(output)
        return output


class ParallelTransformerDecoder(MegatronModule):

    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        init_method=init.xavier_uniform_,
        ):

        super(ParallelTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([])
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = dropout
        self.bias = bias
        self.init_method = init_method
        self.layers.extend([self.build_decoder_layer() for i in
                           range(self.num_layers)])
        self.norm = FusedLayerNorm(self.embed_dim)

    def build_decoder_layer(self):
        layer = DecoderLayer(self.embed_dim, self.num_heads,
                             dropout=self.attn_dropout, bias=self.bias,
                             init_method=self.init_method)
        return layer

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        ):
        """Pass the inputs (and mask) through the decoder layer in turn.
        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Returns:
            decoded output of shape (tgt_len, batch, embed_dim)
        """

        output = tgt
        for mod in self.layers:
            output = mod(output, encoder_out=memory,
                         encoder_padding_mask=memory_key_padding_mask,
                         self_attn_mask=tgt_mask,
                         self_attn_padding_mask=tgt_key_padding_mask)
        output = self.norm(output)
        return output
