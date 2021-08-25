from typing import Optional, Tuple
from apex.normalization import FusedLayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from megatron import mpu
try:
    from megatron.module import MegatronModule # v 1.1.5
except:
    from megatron.model.module import MegatronModule


class MultiheadAttention(MegatronModule):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        cross_attention=False,
        init_method=init.xavier_uniform_,
        ):

        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = nn.Dropout(p=dropout)
        self.bias = bias
        self.cross_attention = cross_attention
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.init_method = init_method
        self.skip_bias = not bias

        # Self-Attention is Column Parallelized
        self.query_key_value = mpu.ColumnParallelLinear(self.embed_dim,
                3 * self.embed_dim, gather_output=True,
                init_method=self.init_method,
                skip_bias_add=self.skip_bias)

        # Cross-Attention is Row and Column Parallelized
        self.q_proj = mpu.RowParallelLinear(self.embed_dim,
                self.embed_dim, input_is_parallel=False,
                init_method=self.init_method, bias=bias,
                skip_bias_add=self.skip_bias)
        self.key_value = mpu.ColumnParallelLinear(self.embed_dim, 2
                * self.embed_dim, gather_output=True,
                init_method=self.init_method,
                skip_bias_add=self.skip_bias)

        # Final projection is Row Parallelized
        self.out_proj = mpu.RowParallelLinear(self.embed_dim,
                self.embed_dim, input_is_parallel=False,
                init_method=self.init_method, bias=bias)

    def forward(
        self,
        query,
        key=None,
        value=None,
        key_padding_mask=None,
        attn_mask=None,
        ):
        """Input shape: Time x Batch x Channel

        Args:
            query - tokens/states of shape [Time x Batch x Channel]
            key - tokens/states of shape [Time x Batch x Channel]
            value - tokens/states of shape [Time x Batch x Channel]
            key_padding_mask - keys that are pads where padding
                elements are indicated by 1s. Shape: [batch, src_len].
            attn_mask - typically used to implement causal attention, where
                the mask prevents the attention from looking forward in time.
                Shape: [tgt_len, src_len].
        Returns:
            outputs - attention probability scores of shape (Time x Batch x Channel)
        """

        (tgt_len, bsz, embed_dim) = query.size()

        # Compute attention projections
        if not self.cross_attention:
            (q_k_v, bias) = self.query_key_value(query)
            (q, k, v) = mpu.split_tensor_along_last_dim(q_k_v, 3)
        else:
            q, _ = self.q_proj(query)
            if key is None:
                assert value is None, \
                    'Cross attention mode: since key is None, value must also be None.'
                k = v = None
            else:
                (k_v, bias) = self.key_value(key)
                (k, v) = mpu.split_tensor_along_last_dim(k_v, 2)

        # Scale query and reshape
        q = q.contiguous()
        q *= self.scaling
        q = q.view(tgt_len, bsz * self.num_heads,
                   self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads,
                                    self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads,
                                    self.head_dim).transpose(0, 1)

        # Compute attention scores
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads,
                tgt_len, src_len]

        # Apply causal attention mask
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        # Apply padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads,
                    tgt_len, src_len)
            attn_weights = \
                attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads,
                    tgt_len, src_len)

        # Compute attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_weights)

        # Compute context and output projection
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len,
                self.head_dim]
        if attn.size(1) == 1:  # a single decoder step (sequence length == 1)
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz,
                    embed_dim)
        (attn, bias) = self.out_proj(attn)
        attn_output_weights = attn_probs.view(bsz, self.num_heads,
                tgt_len, src_len)
        attn_output_weights = attn_output_weights.sum(dim=1) \
            / self.num_heads
        return (attn, attn_output_weights)


class EncoderLayer(MegatronModule):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        init_method=init.xavier_uniform_,
        ):

        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            cross_attention=False,
            init_method=init_method,
            )
        self.self_attn_layer_norm = FusedLayerNorm(embed_dim)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.activation_fn = F.gelu
        self.activation_dropout = nn.Dropout(p=dropout)
        self.fc1 = mpu.ColumnParallelLinear(embed_dim, 4
                * embed_dim, gather_output=False,
                init_method=init_method, skip_bias_add=False)
        self.fc2 = mpu.RowParallelLinear(4 * embed_dim,
                embed_dim, input_is_parallel=True,
                init_method=init_method, skip_bias_add=False)
        self.final_layer_norm = FusedLayerNorm(embed_dim)

    def forward(
        self,
        x,
        encoder_padding_mask=None,
        attn_mask=None,
        ):
        """
        Args:
            x: input to the layer of shape (seq_len, batch, embed_dim)
            encoder_padding_mask: binary ByteTensor of shape
                (batch, seq_len) where padding elements are indicated by 1.
            attn_mask: binary tensor of shape (tgt_len, src_len),
                where tgt_len is the length of output and src_len is the
                length of input, though here both are equal to seq_len.
        Returns:
            encoded output of shape (seq_len, batch, embed_dim)
        """

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool),
                    -1e8)
        residual = x
        x = self.self_attn_layer_norm(x)
        (x, weights) = self.self_attn(query=x, key=x, value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask)
        x = self.attn_dropout(x)
        x = x + residual
        residual = x
        x = self.final_layer_norm(x)
        x, _ = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout(x)
        x, _ = self.fc2(x)
        x = self.attn_dropout(x)
        x = x + residual
        return x


class DecoderLayer(MegatronModule):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        init_method=init.xavier_uniform_,
        ):

        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            cross_attention=False,
            init_method=init_method,
            )
        self.self_attn_layer_norm = FusedLayerNorm(embed_dim)
        self.encoder_attn = MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            cross_attention=True,
            init_method=init_method,
            )
        self.encoder_attn_layer_norm = FusedLayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.activation_fn = F.gelu
        self.activation_dropout = nn.Dropout(p=dropout)
        self.fc1 = mpu.ColumnParallelLinear(embed_dim, 4
                * embed_dim, gather_output=False,
                init_method=init_method, skip_bias_add=False)
        self.fc2 = mpu.RowParallelLinear(4 * embed_dim,
                embed_dim, input_is_parallel=True,
                init_method=init_method, skip_bias_add=False)
        self.final_layer_norm = FusedLayerNorm(embed_dim)

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        ):
        """
        Args:
            x: input to decoder layer of shape (seq_len, batch, embed_dim)
            encoder_out: output from the encoder
            encoder_padding_mask: binary ByteTensor of shape
                (batch, seq_len) where padding elements are indicated by 1
            self_attn_mask: binary tensor of shape (tgt_len, src_len),
                where tgt_lent is the length of output and src_len is the
                length of input, though here both are equal to seq_len.
            self_attn_padding_mask: binary ByteTensor of shape
                (batch, seq_len) where padding elements are indicated by 1.
        Returns:
            encoded output of shape (seq_len, batch, embed_dim)
        """
        residual = x
        x = self.self_attn_layer_norm(x)

        # Self-Attention block
        (x, weights) = self.self_attn(query=x, key=x, value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask)
        x = self.dropout(x)
        x = x + residual

        # Cross-Attention block
        if encoder_out is not None:
            residual = x
            x = self.encoder_attn_layer_norm(x)
            (x, attn) = self.encoder_attn(query=x, key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask)
            x = self.dropout(x)
            x = x + residual
        residual = x
        x = self.final_layer_norm(x)

        # Fully-connected block
        x, _ = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout(x)
        x, _ = self.fc2(x)
        x = self.dropout(x)
        x = x + residual
        return x
