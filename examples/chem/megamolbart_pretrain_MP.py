# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from dataclasses import asdict
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.plugins.environments.torchelastic_environment import TorchElasticEnvironment
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

# from nemo.collections.nlp.models.language_modeling.megatron_bart_model import MegatronBARTModel
from nemo.collections.nlp.parts.nlp_overrides import GradScaler, MegatronHalfPrecisionPlugin, NLPDDPPlugin
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager

from nemo_chem.models.megamolbart_MP import MegaMolBARTModel
from nemo_chem.data import MoleculeCsvDatasetConfig
from preprocess import Preprocess


def recursive_make_dirs(directory):
    logging.info(f'Creating directory {str(directory)}...')
    if isinstance(directory, str):
        directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)


def update_dataset_config(cfg):
    with open_dict(cfg):
        dataset_cfg = asdict(MoleculeCsvDatasetConfig())
        dataset_cfg.update(cfg.model['data'])
        cfg.model['data'] = dataset_cfg
    return cfg


def setup_trainer(cfg):
    megatron_amp_o2 = cfg.model.get('megatron_amp_O2', False)
    plugins = [
        NLPDDPPlugin(
            no_ddp_communication_hook=(
                megatron_amp_o2 and cfg.trainer.precision == 'bf16'
            ),  # Only bf16 uses fp32_grad_accum.
            gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
            find_unused_parameters=False,
        )
    ]
    if cfg.trainer.precision in [16, 'bf16']:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
        if megatron_amp_o2:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
        else:
            plugins.append(NativeMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, **cfg.trainer, callbacks=[ModelSummary(max_depth=3)])

    resume_from_checkpoint = trainer.checkpoint_connector.resume_from_checkpoint_fit_path
    trainer.checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)
    
    # Override timer callback to a stateless one
    for idx, callback in enumerate(trainer.callbacks):
        if isinstance(callback, Timer):
            trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time,)

    return trainer


@hydra_runner(config_path="conf", config_name="megamolbart_pretrain_tiny_MP")
def main(cfg) -> None:
    cfg = update_dataset_config(cfg)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = setup_trainer(cfg)

    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    recursive_make_dirs(log_dir)
    recursive_make_dirs(trainer.checkpoint_callback.dirpath)

    # update resume from checkpoint found by exp_manager
    resume_from_checkpoint = trainer.checkpoint_connector.resume_from_checkpoint_fit_path
    logging.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    model = MegaMolBARTModel(cfg.model, trainer)

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
