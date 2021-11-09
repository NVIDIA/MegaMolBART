from typing import Optional, Union, Any, List
from copy import deepcopy
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from preprocess import Preprocess

from nemo.utils import logging
from nemo.utils.config_utils import update_model_config
from nemo.core.config import hydra_runner
from nemo.core.config.modelPT import NemoConfig
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils.exp_manager import exp_manager, ExpManagerConfig

from nemo.collections.chem.models import MegaMolBARTModel, MegatronBARTConfig
from nemo.collections.chem.tokenizer import MolEncTokenizerFromVocabFileConfig
from nemo.collections.chem.decoder import DecodeSamplerConfig


@dataclass
class MegaMolBARTPretrain(NemoConfig):
    name: Optional[str] = 'MegaMolBART'
    do_training: Optional[bool] = None
    do_testing: Optional[bool] = None
    model: MegatronBARTConfig = MegatronBARTConfig()
    tokenizer: MolEncTokenizerFromVocabFileConfig = MolEncTokenizerFromVocabFileConfig()
    trainer: Optional[TrainerConfig] = TrainerConfig()
    exp_manager: Optional[ExpManagerConfig] = ExpManagerConfig(name='MegaMolBART', files_to_copy=[])
    seed: Optional[int] = None
    dataset_path: Optional[str] = None

@hydra_runner()
def main(cfg: MegaMolBARTPretrain) -> None:

    # Load configuration
    default_cfg = OmegaConf.structured(MegaMolBARTPretrain())
    OmegaConf.set_struct(cfg, False)
    cfg = update_model_config(default_cfg, cfg)
    OmegaConf.set_struct(cfg, True)

    logging.info("************** Experiment configuration ***********")
    logging.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")

    # Make dict from trainer to add DDPPlugin because typechecking it is a nightmare
    trainer_config = dict(deepcopy(cfg.trainer))
    trainer_config['plugins'] = [DDPPlugin(find_unused_parameters=True)]

    if cfg.seed:
        pl.seed_everything(cfg.seed, workers=True)

    if trainer_config['precision'] != 'bf16':
        trainer_config['precision'] = int(trainer_config['precision']) # TODO figure out why this is recognized as a string
    trainer = pl.Trainer(**trainer_config)
    exp_manager(trainer, cfg.get("exp_manager", None))

    model = MegaMolBARTModel(cfg, trainer)
    logging.info("************** Model parameters and their sizes ***********")
    for name, param in model.named_parameters():
        logging.info(f'{name}: {param.size()}')
    logging.info("***********************************************************")

    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        trainer.fit(model)
        logging.info("************** Finished Training ***********")
    else:
        logging.info("************** Starting Data PreProcessing ***********")
        preprocess = Preprocess()
        preprocess.split_dataset(links_file='conf/model/dataset/ZINC-downloader-small.txt',
                                 output_dir=cfg.dataset_path)
        logging.info("************** Finished Data PreProcessing ***********")

    if cfg.do_testing:
        logging.info("************** Starting Testing ***********")
        trainer.test(model)
        logging.info("************** Finished Testing ***********")


if __name__ == '__main__':

    main()


