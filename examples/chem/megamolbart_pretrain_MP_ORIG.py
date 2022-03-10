from dataclasses import dataclass, asdict
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.plugins.environments.torchelastic_environment import TorchElasticEnvironment
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.callbacks.timer import Timer

from nemo.utils import logging
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager, ExpManagerConfig, StatelessTimer
from nemo.core.config.modelPT import NemoConfig
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.collections.nlp.parts.nlp_overrides import GradScaler, NLPDDPPlugin
from nemo.utils.config_utils import update_model_config

from nemo_chem.tokenizer import MolEncTokenizerFromVocabFileConfig
from nemo_chem.decoder import DecodeSamplerConfig
from nemo_chem.models.megamolbart_MP import MegaMolBARTModel
from nemo_chem.data import MoleculeCsvDatasetConfig
from preprocess import Preprocess

# @dataclass
# class MegaMolBARTPretrain(NemoConfig):
#     name: Optional[str] = 'MegaMolBART'
#     do_training: Optional[bool] = None
#     do_testing: Optional[bool] = None
#     restore_from_path: Optional[str] = None
#     trainer: Optional[TrainerConfig] = TrainerConfig()
#     exp_manager: Optional[ExpManagerConfig] = ExpManagerConfig(name='MegaMolBART', files_to_copy=[])
#     seed: Optional[int] = None
#     dataset_path: Optional[str] = None
#     model: MegatronBARTConfig = MegatronBARTConfig()
#     tokenizer: MolEncTokenizerFromVocabFileConfig = MolEncTokenizerFromVocabFileConfig()


def recursive_make_dirs(directory):
    logging.info(f'Creating directory {str(directory)}...')
    if isinstance(directory, str):
        directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)


def update_dataset_config(cfg):
    # TODO find a more elegant way to enforce this
    with open_dict(cfg):
        train_cfg = asdict(MoleculeCsvDatasetConfig())
        train_cfg.update(cfg.model['train_dataset'])
        cfg.model['train_dataset'] = train_cfg

        if cfg.model.get('validation_dataset', False):
            val_cfg = asdict(MoleculeCsvDatasetConfig())
            val_cfg.update(cfg.model['validation_dataset'])
            cfg.model['validation_dataset'] = val_cfg

        if cfg.model.get('test_dataset', False):
            test_cfg = asdict(MoleculeCsvDatasetConfig())
            test_cfg.update(cfg.model['test_dataset'])
            cfg.model['test_dataset'] = test_cfg

    return cfg


@hydra_runner(config_path="conf", config_name="megamolbart_pretrain_tiny_span_aug_MP")
def main(cfg) -> None:
    cfg = update_dataset_config(cfg)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    plugins = [
        NLPDDPPlugin() # TODO update when pipeline parallel is supported
    ]
    if cfg.trainer.precision in [16, 'bf16']:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
        # TODO update when pipeline parallel is supported
        plugins.append(NativeMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
            

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    if cfg.seed:
        pl.seed_everything(cfg.seed, workers=True)

    trainer = Trainer(plugins=plugins, **cfg.trainer, callbacks=[ModelSummary(max_depth=3)])

    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    recursive_make_dirs(log_dir)
    recursive_make_dirs(trainer.checkpoint_callback.dirpath)

    # update resume from checkpoint found by exp_manager
    resume_from_checkpoint = trainer.checkpoint_connector.resume_from_checkpoint_fit_path
    logging.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')

    trainer.checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)
    # Override timer callback to a stateless one
    for idx, callback in enumerate(trainer.callbacks):
        if isinstance(callback, Timer):
            trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time,)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

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
