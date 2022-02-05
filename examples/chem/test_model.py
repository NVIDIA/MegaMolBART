import os
import hydra
from omegaconf import DictConfig, OmegaConf
from nemo.utils import logging
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.collections.nlp.models.language_modeling.megatron.tokens_encoder_decoder import TokensEncoderDecoderModule

from nemo_chem.decoder import DecodeSampler
from nemo_chem.tokenizer import MolEncTokenizer, VOCAB_DIR
from nemo_chem.models.megamolbart.megatron_bart_base import MegatronBART

from IPython import embed

def initialize_megatron():
    initialize_model_parallel_for_nemo(
        world_size=1,
        global_rank=0,
        local_rank=0,
        tensor_model_parallel_size=1,
        seed=1234,
    )

def cfg_cleanup(cfg):
    pop_list = ['train_ds', 'validation_ds', 'test_ds', 'name', 'optim', 'tensor_model_parallel_size', 'pretrained', 'checkpoint_file']
    for pop_name in pop_list:
        cfg.pop(pop_name, None)

    cfg['vocab_size'] = 128
    return cfg

def print_param_sizes(model):
    logging.info("************** Model parameters and their sizes ***********")
    for name, param in model.named_parameters():
        logging.info(f'{name}: {param.size()}')
    logging.info("***********************************************************")


@hydra.main(config_path="conf", config_name="megamolbart_pretrain_small_span_aug_MP")
def load_nemo(cfg):
    cfg_model = dict(cfg.model)
    cfg_model = cfg_cleanup(cfg_model)
    cfg_model.pop('seq_len', None)

    logging.info(OmegaConf.to_yaml(cfg_model))
    model = TokensEncoderDecoderModule(**cfg_model)
    print_param_sizes(model)
    return 


@hydra.main(config_path="conf", config_name="megamolbart_pretrain_small_span_aug")
def load_orig(cfg):
    cfg_model = dict(cfg.model)
    cfg_model = cfg_cleanup(cfg_model)
    cfg_model['pad_token_idx'] = 0
    
    logging.info(OmegaConf.to_yaml(cfg_model))
    tokenizer = tokenizer = MolEncTokenizer.from_vocab_file(os.path.join(VOCAB_DIR, 'megamolbart_vocab.txt'))
    cfg_model['decode_sampler'] = DecodeSampler(tokenizer, cfg_model['seq_len'])

    model = MegatronBART(**cfg_model)
    print_param_sizes(model)
    return 





if __name__ == '__main__':
    initialize_megatron()
    load_orig()
    load_nemo()
    