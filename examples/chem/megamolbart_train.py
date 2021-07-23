import sys
sys.path.insert(0, '/workspace') # TODO remove this
sys.path.insert(0, '/code/NeMo') # TODO remove this

from typing import Optional
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from nemo.utils import logging

from nemo.utils.config_utils import update_model_config
from nemo.core.config.modelPT import NemoConfig
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils.exp_manager import exp_manager, ExpManagerConfig

from nemo.collections.chem.models import MegaMolBARTModel, MegatronBARTConfig
from nemo.collections.chem.tokenizer import MolEncTokenizerFromVocabFileConfig
from nemo.collections.chem.decoder import DecodeSamplerConfig
from nemo.collections.chem.data import MoleculeDatasetConfig

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
    pl.seed_everything(seed, workers=True)

    # Load configuration
    cfg = OmegaConf.load('/code/NeMo/examples/chem/conf/megamolbart_base.yaml')    
    
    OmegaConf.set_struct(cfg, False)
    default_cfg = OmegaConf.structured(MegaMolBARTTrain())
    cfg = update_model_config(default_cfg, cfg)
    OmegaConf.set_struct(cfg, True)

    cfg = DictConfig(cfg)
    cfg['trainer']['plugins'] = []
    cfg['trainer']['limit_train_batches'] = 10
    cfg['trainer']['limit_val_batches'] = 2
    cfg['trainer']['limit_test_batches'] = 2

    logging.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    trainer = pl.Trainer(**cfg.trainer)

    exp_manager(trainer, cfg.get("exp_manager", None))
    model = MegaMolBARTModel(cfg, trainer)
    trainer.fit(model)

    