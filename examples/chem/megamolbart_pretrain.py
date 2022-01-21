from typing import Optional, Union, Any, List
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.trainer.trainer import Trainer
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments.torchelastic_environment import TorchElasticEnvironment
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.callbacks.timer import Timer
from preprocess import Preprocess

from nemo.utils import logging
from nemo.utils.config_utils import update_model_config
from nemo.core.config import hydra_runner
from nemo.core.config.modelPT import NemoConfig
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils.exp_manager import exp_manager, ExpManagerConfig, StatelessTimer
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.parts.nlp_overrides import GradScaler, NLPDDPPlugin

from nemo_chem.models import MegaMolBARTModel, MegatronBARTConfig
from nemo_chem.tokenizer import MolEncTokenizerFromVocabFileConfig
from nemo_chem.decoder import DecodeSamplerConfig
from nemo_chem.nlp_overrides import MegaMolBARTNLPDDPPlugin



def recursive_make_dirs(directory):
    logging.info(f'Creating directory {str(directory)}...')
    if isinstance(directory, str):
        directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)


def configure_trainer_plugins(cfg: DictConfig) -> DictConfig:
    trainer_cfg = dict(deepcopy(cfg.trainer))
    
    if trainer_cfg['precision'] != 'bf16':
        trainer_cfg['precision'] = int(trainer_cfg['precision'])

    # Configure plugins
    # plugins = [NLPDDPPlugin(num_nodes=trainer_cfg['num_nodes'])]
    plugins = [MegaMolBARTNLPDDPPlugin(num_nodes=trainer_cfg['num_nodes'])] # TODO revert to NLPDDPPlugin when find_unused_parameters bug is solved
    if cfg.trainer.precision == 16:
        scaler = GradScaler(
            init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
            growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
        )
        plugins.append(NativeMixedPrecisionPlugin(precision=16, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    if not trainer_cfg.get('plugins', False):
        trainer_cfg['plugins'] = []
    trainer_cfg['plugins'].extend(plugins)
    return trainer_cfg


def update_checkpoint_name(cfg: DictConfig, trainer: Trainer):
    "Update config checkpoint name for model parallel if needed"
    resume_from_checkpoint = trainer.resume_from_checkpoint
    if resume_from_checkpoint is not None:
        mp_rank = compute_model_parallel_rank(trainer.local_rank, cfg.model.tensor_model_parallel_size)
        resume_from_checkpoint = Path(resume_from_checkpoint)
        resume_from_checkpoint = resume_from_checkpoint.parent.parent.joinpath(f'mp_rank_{mp_rank:02d}').joinpath(
            resume_from_checkpoint.name
        )
        resume_from_checkpoint = str(resume_from_checkpoint)

        logging.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')
        trainer.checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)
    return trainer


def use_stateless_timer(cfg: DictConfig, trainer: Trainer):
    for idx, callback in enumerate(trainer.callbacks):
        if isinstance(callback, Timer):
            trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time)

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

    trainer_config = configure_trainer_plugins(cfg)
    if cfg.seed:
        pl.seed_everything(cfg.seed, workers=True)

    trainer = pl.Trainer(**trainer_config)
    trainer = update_checkpoint_name(cfg, trainer)
    use_stateless_timer(cfg, trainer)

    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    recursive_make_dirs(log_dir)
    recursive_make_dirs(trainer.checkpoint_callback.dirpath)

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
