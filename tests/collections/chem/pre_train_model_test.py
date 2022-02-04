# coding=utf-8

import pytest
import random
import torch

from nemo_chem.decoder import DecodeSampler
from nemo_chem.tokenizer import MolEncTokenizer, MolEncTokenizerFromSmilesConfig
from nemo_chem.models import MegatronBART, MegatronBARTConfig

from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo

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

random.seed(a=1)
torch.manual_seed(1)

initialize_model_parallel_for_nemo(
    world_size=1,
    global_rank=0,
    local_rank=0,
    tensor_model_parallel_size=1,
    seed=1234,
)

TEST_MODEL_CONFIG = MegatronBARTConfig()
TEST_PERCEIVER_CONFIG = MegatronBARTConfig(encoder_type='perceiver')

@pytest.fixture(params=[TEST_MODEL_CONFIG, TEST_PERCEIVER_CONFIG])
def args(request):
    _args = request.param
    return _args

@pytest.fixture
def tokenizer():
    cfg = MolEncTokenizerFromSmilesConfig({'smiles': react_data + prod_data})
    _tokenizer = MolEncTokenizer.from_smiles(
        cfg.smiles["smiles"], cfg.regex, mask_scheme="replace")
    return _tokenizer

@pytest.fixture
def sampler(args, tokenizer):
    _sampler = DecodeSampler(tokenizer, args.seq_length)
    return _sampler

@pytest.fixture
def model(args, tokenizer, sampler):
    pad_token_idx = tokenizer.vocab[tokenizer.pad_token]
    vocab_size = len(tokenizer)
    _model = MegatronBART(sampler,
                          args.encoder_type,
                          pad_token_idx,
                          vocab_size,
                          args.blocks_model,
                          args.steps_model,
                          args.d_model,
                          args.num_layers,
                          args.num_heads,
                          args.d_feedforward,
                          args.seq_length,
                          dropout=0.1)
    return _model.cuda()

def test_pos_emb_shape(model, sampler, tokenizer, args):
    pos_embs = model._positional_embs()

    assert pos_embs.shape[0] == args.seq_length
    assert pos_embs.shape[1] == model.d_model  # hidden size

def test_construct_input_shape(model, sampler, tokenizer, args):
    token_output = tokenizer.tokenize(react_data, sents2=prod_data, pad=True)
    tokens = token_output["original_tokens"]
    sent_masks = token_output["sentence_masks"]

    token_ids = torch.tensor(
        tokenizer.convert_tokens_to_ids(tokens)).transpose(0, 1).cuda()
    sent_masks = torch.tensor(sent_masks).transpose(0, 1).cuda()

    emb = model._construct_input(token_ids, sent_masks)

    assert emb.shape[0] == max([len(ts) for ts in tokens])
    assert emb.shape[1] == 3
    assert emb.shape[2] == args.d_model

def test_bart_forward_shape(model, sampler, tokenizer, args):
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
    exp_dim = args.d_model  # hidden_size
    exp_vocab_size = len(tokenizer)

    assert tuple(model_output.shape) == (exp_seq_len, exp_batch_size, exp_dim)
    assert tuple(token_output.shape) == (exp_seq_len, exp_batch_size, exp_vocab_size)

def test_bart_encode_shape(model, sampler, tokenizer, args):
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
    if args.encoder_type == 'seq2seq':
        exp_seq_len = 9  # From expected tokenized length of react data
    elif args.encoder_type == 'perceiver':
        exp_seq_len = args.steps_model # From expected num_hidden_steps of the Perceiver encoder
    exp_batch_size = len(react_data)
    exp_dim = args.d_model  # hidden_size

    assert tuple(output.shape) == (exp_seq_len, exp_batch_size, exp_dim)

def test_bart_decode_shape(model, sampler, tokenizer, args):
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
    if args.encoder_type == "perceiver":
        react_mask = torch.zeros(
            (memory.shape[0:2]), dtype=react_mask.dtype, device=react_mask.device)
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

def test_calc_char_acc(model, sampler, tokenizer, args):
    react_token_output = tokenizer.tokenize(react_data[1:], pad=True)
    react_tokens = react_token_output["original_tokens"]
    react_pad_mask = react_token_output["original_pad_masks"]
    target_ids = torch.tensor(
        tokenizer.convert_tokens_to_ids(react_tokens)).T[1:, :]
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
