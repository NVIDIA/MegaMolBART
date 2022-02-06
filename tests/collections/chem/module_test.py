import os
import pytest
import random
import torch
from omegaconf import OmegaConf
from nemo_chem.modules.config import MegatronBARTModuleConfig
from nemo.collections.nlp.models.language_modeling.megatron.tokens_encoder_decoder import TokensEncoderDecoderModule
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo_chem.tokenizer import MolEncTokenizer, VOCAB_DIR

random.seed(a=1)
torch.manual_seed(1)

initialize_model_parallel_for_nemo(
    world_size=1,
    global_rank=0,
    local_rank=0,
    tensor_model_parallel_size=1,
    seed=1,
)

TEST_MODEL_CONFIG = MegatronBARTModuleConfig()
@pytest.fixture(params=[TEST_MODEL_CONFIG])
def args(request):
    _args = OmegaConf.structured(request.param)
    _args = dict(_args)
    for var_remove in ['name', 'tensor_model_parallel_size']:
        _args.pop(var_remove, None)
    return _args

@pytest.fixture
def tokenizer(args):
    vocab_file = os.path.join(VOCAB_DIR, 'megamolbart_vocab.txt')
    _tokenizer = MolEncTokenizer.from_vocab_file(vocab_file)
    return _tokenizer

@pytest.fixture
def model(args, tokenizer):
    args['vocab_size'] = len(tokenizer)
    _model = TokensEncoderDecoderModule( **args)
    return _model.cuda()

def get_num_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn *= s
        pp += nn
    return pp

def get_named_params(model):
    return list(model.named_parameters())

def test_model_size(model, args):
    num_params = get_num_params(model)
    assert num_params == 112807

    layers = get_named_params(model)
    first_layer_name, first_layer_shape = layers[0][0], layers[0][1].shape
    last_layer_name, last_layer_shape = layers[-1][0], layers[-1][1].shape
    assert first_layer_name == 'encoder_embedding.word_embeddings.weight'
    assert first_layer_shape[0] == 523
    assert first_layer_shape[1] == 36
    assert last_layer_name == 'tokens_head.bias'
    assert last_layer_shape[0] == 523

    num_layers = len(layers)
    assert num_layers == 71
    