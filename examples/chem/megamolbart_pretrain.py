from typing import Optional, Union, Any, List
from copy import deepcopy
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from nemo.utils import logging
from nemo.utils.config_utils import update_model_config
from nemo.core.config.modelPT import NemoConfig
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils.exp_manager import exp_manager, ExpManagerConfig

from nemo.collections.chem.models import MegaMolBARTModel, MegatronBARTConfig
from nemo.collections.chem.tokenizer import MolEncTokenizerFromVocabFileConfig
from nemo.collections.chem.decoder import DecodeSamplerConfig

@dataclass
class MegaMolBARTTrain(NemoConfig):
    name: Optional[str] = 'MegaMolBART'
    do_training: bool = False
    do_testing: bool = False
    model: MegatronBARTConfig = MegatronBARTConfig()
    tokenizer: MolEncTokenizerFromVocabFileConfig = MolEncTokenizerFromVocabFileConfig()
    trainer: Optional[TrainerConfig] = TrainerConfig()
    exp_manager: Optional[ExpManagerConfig] = ExpManagerConfig(name='MegaMolBART', files_to_copy=[])

from IPython import embed

if __name__ == '__main__':

    seed = 42
    ngpus = torch.cuda.device_count()
    config_path = f'/code/examples/chem/conf/megamolbart_pretrain.yaml'

    pl.seed_everything(seed, workers=True)

    # Load configuration
    cfg = OmegaConf.load(config_path) 
    OmegaConf.set_struct(cfg, False)
    default_cfg = OmegaConf.structured(MegaMolBARTTrain())
    cfg = update_model_config(default_cfg, cfg)

    cfg.trainer['gpus'] = ngpus
    logging.info(f'Using {ngpus} GPUs ...')

    if ngpus == 2:
        cfg.trainer['limit_train_batches'] = 10
        cfg.trainer['limit_val_batches'] = 2
        cfg.trainer['limit_test_batches'] = 2
    OmegaConf.set_struct(cfg, True)
    logging.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")

    # Make a dict from trainer to add DDPPlugin because struct typechecking is a nightmare
    trainer_config = dict(deepcopy(cfg.trainer))
    trainer_config['plugins'] = [DDPPlugin(find_unused_parameters=True)]
    trainer = pl.Trainer(**trainer_config)

    exp_manager(trainer, cfg.get("exp_manager", None))
    model = MegaMolBARTModel(cfg, trainer)
    trainer.fit(model)

    
