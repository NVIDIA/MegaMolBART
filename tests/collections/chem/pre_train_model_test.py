# coding=utf-8

import pytest
import torch
import random

from megatron.initialize import initialize_megatron
from megatron import get_args
from megatron_molbart.decoder import DecodeSampler
from megatron_molbart.tokenizer import MolEncTokenizer
from megatron_molbart.megatron_bart import MegatronBART
from util import (DEFAULT_NUM_LAYERS, DEFAULT_D_MODEL, DEFAULT_NUM_HEADS, DEFAULT_VOCAB_PATH, CHECKPOINTS_DIR, DEFAULT_MAX_SEQ_LEN)
from util import REGEX as regex

# Use dummy SMILES strings
react_data = [
    "CCO.C",
    "CCCl",
    "C(=O)CBr"
]

# Use dummy SMILES strings
prod_data = [
    "cc",
    "CCl",
    "CBr"
]

args = {
        'num_layers': DEFAULT_NUM_LAYERS,
        'hidden_size': DEFAULT_D_MODEL,
        'num_attention_heads': DEFAULT_NUM_HEADS,
        'max_position_embeddings': DEFAULT_MAX_SEQ_LEN,
        'tokenizer_type': 'GPT2BPETokenizer',
        'vocab_file': DEFAULT_VOCAB_PATH,
        'load': CHECKPOINTS_DIR
    }

initialize_megatron(args_defaults=args, ignore_unknown_args=True)
args = get_args()

random.seed(a=1)
torch.manual_seed(1)


def build_tokenizer():
    tokenizer = MolEncTokenizer.from_smiles(react_data + prod_data, regex, mask_scheme="replace")
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
