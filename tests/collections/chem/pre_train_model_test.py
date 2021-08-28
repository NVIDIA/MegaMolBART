# coding=utf-8

from dataclasses import dataclass
import pytest
import torch
import random
import tempfile

from megatron.initialize import initialize_megatron
from megatron import get_args
from torch.cuda import init
from megatron_molbart.decoder import DecodeSampler
from megatron.mpu import (
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_model_parallel_group,
    model_parallel_is_initialized,
    set_pipeline_model_parallel_rank,
    set_pipeline_model_parallel_world_size,
)
from megatron_molbart.tokenizer import MolEncTokenizer
from megatron_molbart.megatron_bart import MegatronBART
from util import (DEFAULT_NUM_LAYERS, DEFAULT_D_MODEL, DEFAULT_NUM_HEADS, DEFAULT_VOCAB_PATH, CHECKPOINTS_DIR, DEFAULT_MAX_SEQ_LEN)
from util import REGEX as regex
from nemo.utils.app_state import AppState
import logging
import os

DEFAULT_VOCAB_PATH = "/opt/MegaMolBART/bart_vocab.txt"
CHECKPOINTS_DIR = "/models/checkpoints"

@dataclass
class TestInput:
    """Represents input to a test"""
    def __init__(self, init_dict: dict):
        for key in init_dict:
            setattr(self, key, init_dict[key] )
    
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
# args = {
#         'num_layers': DEFAULT_NUM_LAYERS,
#         'hidden_size': DEFAULT_D_MODEL,
#         'num_attention_heads': DEFAULT_NUM_HEADS,
#         'max_position_embeddings': DEFAULT_MAX_SEQ_LEN,
#         'tokenizer_type': 'GPT2BPETokenizer',
#         'micro_batch_size': 8,
#         'merge_file': True,
#         'encoder_seq_length': 8, 
#         #'vocab_file': DEFAULT_VOCAB_PATH,
#         'load': CHECKPOINTS_DIR
#     }
# initialize_megatron(args_defaults=kwargs, ignore_unknown_args=True)
def _set_app_state(self, cfg):
    app_state = AppState()
    if cfg.trainer is not None:
        app_state._world_size = cfg.trainer.num_nodes * cfg.trainer.gpus
        num_gpus = cfg.trainer.gpus
    else:
        app_state._world_size = 1
        num_gpus = 1

    env = os.environ.copy()
    app_state.local_rank = int(env.get('LOCAL_RANK', 0))
    app_state.node_rank = int(env.get('NODE_RANK', 0))
    app_state.global_rank = app_state.local_rank + (app_state.node_rank * num_gpus) # TODO better way to calculate?
    app_state.model_parallel_size = None
    app_state.model_parallel_rank = None
    # app_state.device_id = None # TODO add these
    # app_state.model_parallel_group = None
    # app_state.data_parallel_size = None
    # app_state.data_parallel_rank = None
    # app_state.data_parallel_group = None
    self._app_state = app_state

def _set_ddp(self, trainer):
    # Sampler is replaced manually below because PTL seems to use global_rank instead of local_rank
    if trainer.accelerator_connector.replace_sampler_ddp & (trainer.accelerator_connector.distributed_backend == 'ddp'):
        self.replace_sampler_ddp = True
        trainer.accelerator_connector.replace_sampler_ddp = False
    else:
        self.replace_sampler_ddp = False

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

def setup_megatron() -> dict:
    """Initialize Megatron"""
    app_state = AppState()
    model_parallel_size = app_state.model_parallel_size
    model_parallel_rank = app_state.model_parallel_rank

    # Configure globals
    set_pipeline_model_parallel_rank(0)  # Pipeline model parallelism not currently implemented in NeMo
    set_pipeline_model_parallel_world_size(1)  # Pipeline model parallelism not currently implemented in NeMo

    # megatron input arguments
    args = {'num_layers': DEFAULT_NUM_LAYERS,
            'hidden_size': DEFAULT_D_MODEL,
            'num_attention_heads': DEFAULT_NUM_HEADS,
            'max_position_embeddings': DEFAULT_MAX_SEQ_LEN,
            'onnx_safe': True,
            'lazy_mpu_init': True,
            'tokenizer_type': 'BertWordPieceCase',
            'micro_batch_size': 8,
            'merge_file': True,
            'encoder_seq_length': 8,
            'vocab_file': _get_megatron_vocab_file()}
            # TODO vocab size may need to be set

    # extra args provider
    if model_parallel_size is not None:
        app_state = AppState()
        os.environ["WORLD_SIZE"] = str(app_state.world_size) # Must be set for model parallel megatron-lm
        os.environ["RANK"] = str(model_parallel_rank)
        extra_args_provider = _update_megatron_args(tensor_model_parallel_size=model_parallel_size)
    else:
        extra_args_provider = _update_megatron_args()

    # Initialize part of Megatron global state that is needed for its constructor.
    # We set 'lazy_mpu_init' flag on to make Megatron do only the initialization that does not depend
    # on ddp be initialized yet (and we don't want Megatron to initialize DDP itself either)
    # and to return a hook for us to call after PTL has torch.distributed initialized.
    # (or if no PTL in case of inference - then we'll initialize torch.distributed)
    # We call and clear this hook on first call to forward()
    _lazy_init_fn = initialize_megatron(
        extra_args_provider=extra_args_provider, args_defaults=args, ignore_unknown_args=True
    )
    return args
args = TestInput(setup_megatron())

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
