# coding=utf-8

from dataclasses import dataclass
import pytest
import torch
import random
import tempfile
import logging
import os
from pathlib import Path

from torch.cuda import init
from megatron.initialize import initialize_megatron
from megatron import get_args
from megatron.mpu import set_pipeline_model_parallel_rank, set_pipeline_model_parallel_world_size

from nemo.collections.chem.decoder import DecodeSampler
from nemo.collections.chem.tokenizer import MolEncTokenizer, MolEncTokenizerFromSmilesConfig
from nemo.collections.chem.models import MegatronBART, MegatronBARTConfig

# Use dummy SMILES strings
react_data = [
    "CCO.C",
    "CCCl",
    "C(=O)CBr"
]

prod_data = [
    "cc",
    "CCl",
    "CBr"
]

def update_megatron_args(
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

def get_megatron_vocab_file() -> str:
    """Generate fake Megatron vocab file with required tokens"""
    fake_vocab_contents = '\n'.join(['[CLS]', '[SEP]', '[PAD]', '[MASK]'])
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as fh:
        fh.write(fake_vocab_contents)
        vocab_file = fh.name
    return vocab_file

def setup_megatron() -> dict:
    """Initialize Megatron"""

    # Configure globals
    set_pipeline_model_parallel_rank(0)
    set_pipeline_model_parallel_world_size(1)
    cfg = MegatronBARTConfig()

    # megatron input arguments
    args = {'num_layers': cfg.num_layers,
            'hidden_size': cfg.d_model,
            'num_attention_heads': cfg.num_heads,
            'max_position_embeddings': cfg.max_seq_len,
            'onnx_safe': True,
            'lazy_mpu_init': True,
            'tokenizer_type': 'BertWordPieceCase',
            'micro_batch_size': 8,
            'merge_file': True,
            'vocab_file': get_megatron_vocab_file()}
            # 'encoder_seq_length': 8,
    extra_args_provider = update_megatron_args()
    
    initialize_megatron(extra_args_provider=extra_args_provider, args_defaults=args, ignore_unknown_args=True)
    return args

@dataclass
class TestInput:
    """Represents input to a test"""
    def __init__(self, init_dict: dict):
        for key in init_dict:
            setattr(self, key, init_dict[key] )

args = TestInput(setup_megatron())

random.seed(a=1)
torch.manual_seed(1)

def build_tokenizer():
    cfg = MolEncTokenizerFromSmilesConfig({'smiles': react_data + prod_data})
    tokenizer = MolEncTokenizer.from_smiles(cfg.smiles, cfg.regex, mask_scheme="replace")
    return tokenizer

def build_sampler(args, tokenizer):
    sampler = DecodeSampler(tokenizer, args.max_position_embeddings)
    return sampler

def build_model(args, tokenzier, sampler):
    pad_token_idx = tokenzier.vocab[tokenzier.pad_token]  
    model = MegatronBART(sampler,
                        pad_token_idx,
                        len(tokenzier),
                        args.hidden_size,
                        args.num_layers,
                        args.num_attention_heads,
                        args.hidden_size * 4,
                        args.max_position_embeddings,
                        dropout=0.1)
    return model.cuda()

def test_pos_emb_shape():
    tokenizer = build_tokenizer()
    sampler = build_sampler(args, tokenizer)
    model = build_model(args, tokenizer, sampler)

    pos_embs = model._positional_embs()

    assert pos_embs.shape[0] == args.max_position_embeddings
    assert pos_embs.shape[1] == model.d_model # hidden size


def test_construct_input_shape():
    tokenizer = build_tokenizer()
    sampler = build_sampler(args, tokenizer)
    model = build_model(args, tokenizer, sampler)

    token_output = tokenizer.tokenize(react_data, sents2=prod_data, pad=True)
    tokens = token_output["original_tokens"]
    sent_masks = token_output["sentence_masks"]

    token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).transpose(0, 1).cuda()
    sent_masks = torch.tensor(sent_masks).transpose(0, 1).cuda()

    emb = model._construct_input(token_ids, sent_masks)

    assert emb.shape[0] == max([len(ts) for ts in tokens])
    assert emb.shape[1] == 3
    assert emb.shape[2] == args.hidden_size


def test_bart_forward_shape():
    tokenizer = build_tokenizer()
    sampler = build_sampler(args, tokenizer)
    model = build_model(args, tokenizer, sampler)

    react_token_output = tokenizer.tokenize(react_data, mask=True, pad=True)
    react_tokens = react_token_output["masked_tokens"]
    react_pad_mask = react_token_output["masked_pad_masks"]
    react_ids = torch.tensor(tokenizer.convert_tokens_to_ids(react_tokens)).T
    react_mask = torch.tensor(react_pad_mask).T

    prod_token_output = tokenizer.tokenize(prod_data, pad=True)
    prod_tokens = prod_token_output["original_tokens"]
    prod_pad_mask = prod_token_output["original_pad_masks"]
    prod_ids = torch.tensor(tokenizer.convert_tokens_to_ids(prod_tokens)).T
    prod_mask = torch.tensor(prod_pad_mask).T

    batch_input = {
        "encoder_input": react_ids.cuda(),
        "encoder_pad_mask": react_mask.cuda(),
        "decoder_input": prod_ids.cuda(),
        "decoder_pad_mask": prod_mask.cuda()
    }

    output = model(batch_input)
    model_output = output["model_output"]
    token_output = output["token_output"]

    exp_seq_len = 4  # From expected tokenized length of prod data
    exp_batch_size = len(prod_data)
    exp_dim = args.hidden_size
    exp_vocab_size = len(tokenizer)

    assert tuple(model_output.shape) == (exp_seq_len, exp_batch_size, exp_dim)
    assert tuple(token_output.shape) == (exp_seq_len, exp_batch_size, exp_vocab_size)


def test_bart_encode_shape():
    tokenizer = build_tokenizer()
    sampler = build_sampler(args, tokenizer)
    model = build_model(args, tokenizer, sampler)

    react_token_output = tokenizer.tokenize(react_data, mask=True, pad=True)
    react_tokens = react_token_output["masked_tokens"]
    react_pad_mask = react_token_output["masked_pad_masks"]
    react_ids = torch.tensor(tokenizer.convert_tokens_to_ids(react_tokens)).T
    react_mask = torch.tensor(react_pad_mask).T

    batch_input = {
        "encoder_input": react_ids.cuda(),
        "encoder_pad_mask": react_mask.cuda()
    }

    output = model.encode(batch_input)

    exp_seq_len = 9  # From expected tokenized length of react data
    exp_batch_size = len(react_data)
    exp_dim = args.hidden_size

    assert tuple(output.shape) == (exp_seq_len, exp_batch_size, exp_dim)


def test_bart_decode_shape():
    tokenizer = build_tokenizer()
    sampler = build_sampler(args, tokenizer)
    model = build_model(args, tokenizer, sampler)

    react_token_output = tokenizer.tokenize(react_data, mask=True, pad=True)
    react_tokens = react_token_output["masked_tokens"]
    react_pad_mask = react_token_output["masked_pad_masks"]
    react_ids = torch.tensor(tokenizer.convert_tokens_to_ids(react_tokens)).T
    react_mask = torch.tensor(react_pad_mask).T

    encode_input = {
        "encoder_input": react_ids.cuda(),
        "encoder_pad_mask": react_mask.cuda()
    }
    memory = model.encode(encode_input)

    prod_token_output = tokenizer.tokenize(prod_data, pad=True)
    prod_tokens = prod_token_output["original_tokens"]
    prod_pad_mask = prod_token_output["original_pad_masks"]
    prod_ids = torch.tensor(tokenizer.convert_tokens_to_ids(prod_tokens)).T
    prod_mask = torch.tensor(prod_pad_mask).T

    batch_input = {
        "decoder_input": prod_ids.cuda(),
        "decoder_pad_mask": prod_mask.cuda(),
        "memory_input": memory.cuda(),
        "memory_pad_mask": react_mask.cuda()
    }
    output = model.decode(batch_input)

    exp_seq_len = 4  # From expected tokenized length of prod data
    exp_batch_size = len(react_data)
    exp_vocab_size = len(tokenizer)

    assert tuple(output.shape) == (exp_seq_len, exp_batch_size, exp_vocab_size)


@pytest.mark.skip(reason="Currently failing due to dimension issue.")
def test_calc_char_acc():
    tokenizer = build_tokenizer()
    sampler = build_sampler(args, tokenizer)
    model = build_model(args, tokenizer, sampler)

    react_token_output = tokenizer.tokenize(react_data[1:], pad=True)
    react_tokens = react_token_output["original_tokens"]
    react_pad_mask = react_token_output["original_pad_masks"]
    target_ids = torch.tensor(tokenizer.convert_tokens_to_ids(react_tokens)).T[1:, :]
    target_mask = torch.tensor(react_pad_mask).T[1:, :]

    # 9 is expected seq len of react data when padded
    token_output = torch.rand([8, len(react_data[1:]), len(tokenizer)])

    """
    Expected outputs 
    CCCl
    C(=O)CBr

    Vocab:
    0 <PAD>
    3 &
    6 C
    7 O
    8 .
    9 Cl
    10 (
    11 =
    12 )
    13 Br
    """

    # Batch element 0
    token_output[0, 0, 6] += 1
    token_output[1, 0, 6] -= 1
    token_output[2, 0, 9] += 1
    token_output[3, 0, 3] += 1
    token_output[4, 0, 0] += 1
    token_output[5, 0, 0] -= 1

    # Batch element 1
    token_output[0, 1, 6] += 1
    token_output[1, 1, 10] += 1
    token_output[2, 1, 11] += 1
    token_output[3, 1, 7] += 1
    token_output[4, 1, 12] -= 1
    token_output[5, 1, 6] += 1
    token_output[6, 1, 13] -= 1
    token_output[7, 1, 3] += 1

    batch_input = {
        "target": target_ids.cuda(),
        "target_pad_mask": target_mask.cuda()
    }

    model_output = {
        "token_output": token_output.cuda()
    }

    token_acc = model._calc_char_acc(batch_input, model_output)
    exp_token_acc = (3 + 6) / (4 + 8)
    assert exp_token_acc == token_acc
