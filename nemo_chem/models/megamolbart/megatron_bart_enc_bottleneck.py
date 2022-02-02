from typing import Optional, Tuple
from apex.normalization import FusedLayerNorm

import torch
import torch.nn as nn
from torch.nn import init

from nemo.collections.nlp.modules.common.megatron.module import MegatronModule

from .megatron_bart_enc_dec import ParallelTransformerEncoder, ParallelTransformerDecoder


class ParallelTransformerEncoderPerceiver(MegatronModule):
    """
    A Perceiver encoder with a fixed-size bottleneck.
    """

    def __init__(
        self,
        num_blocks,
        num_layers,
        num_hidden_steps,
        d_model,
        num_heads,
        dropout=0.0,
        bias=True,
        init_method=init.xavier_uniform_,
    ):

        super(ParallelTransformerEncoderPerceiver, self).__init__()

        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_hidden_steps = num_hidden_steps
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_dropout = dropout
        self.bias = bias
        self.init_method = init_method

        # initial hidden state
        self.init_hidden = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_hidden_steps, d_model)))

        self.self_att_layers = nn.ModuleList([])
        self.self_att_layers.extend([self.build_encoder_layer() for i in range(self.num_blocks)])

        self.cross_att_layers = nn.ModuleList([])
        self.cross_att_layers.extend([self.build_decoder_layer() for i in range(self.num_blocks)])

    def build_encoder_layer(self):
        layer = ParallelTransformerEncoder(
            num_layers=self.num_layers,
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.attn_dropout,
            bias=self.bias,
            init_method=self.init_method,
        )

        return layer

    def build_decoder_layer(self):
        layer = ParallelTransformerDecoder(
            # num_layers=self.num_layers,
            num_layers=1,
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.attn_dropout,
            bias=self.bias,
            init_method=self.init_method,
        )

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
            encoded output of shape (src_len, batch, d_model)
        """
        # initialize hidden state
        hidden = self.init_hidden.unsqueeze(1).expand(-1, src.shape[1], -1)
        hidden_pad_mask = torch.zeros(
            (hidden.shape[0:2]), dtype=src_key_padding_mask.dtype, device=src_key_padding_mask.device).t()

        for cross_att, self_att in zip(self.cross_att_layers, self.self_att_layers):
            residual = hidden

            # cross attention of hidden over encoder states
            hidden = cross_att(
                tgt=hidden,
                memory=src,
                tgt_key_padding_mask=hidden_pad_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

            cross_att(
                tgt=hidden,
                memory=src,
                tgt_key_padding_mask=hidden_pad_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

            # self-attention over hidden
            hidden = self_att(
                src=hidden,
                src_key_padding_mask=hidden_pad_mask,
            )

            # residual connection
            hidden += residual

        return hidden
