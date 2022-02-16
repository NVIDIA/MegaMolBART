from typing import List
import re
import braceexpand
from omegaconf import DictConfig, open_dict
import torch.utils.data as pt_data
from pytorch_lightning.trainer.trainer import Trainer

from nemo.utils import logging
from .csv_data import MoleculeDataset, MoleculeIterableDataset
from .concat import ConcatIterableDataset

__all__ = ['expand_dataset_paths', 'build_train_valid_test_datasets']


def expand_dataset_paths(filepath: str) -> List[str]:
    """Expand dataset paths from braces"""
    # TODO this should go in a Nemo fileutils module or similar
    filepath = re.sub(r"""\(|\[|\<|_OP_""", '{', filepath) # replaces '(', '[', '<' and '_OP_' with '{'
    filepath = re.sub(r"""\)|\]|\>|_CL_""", '}', filepath) # replaces ')', ']', '>' and '_CL_' with '}'
    dataset_paths = list(braceexpand.braceexpand(filepath))
    return dataset_paths


def _build_train_valid_test_datasets(
    cfg: DictConfig, 
    trainer: Trainer
):
    # Setup config
    cfg = cfg.copy()

    with open_dict(cfg):
        filepath = cfg.pop('filepath', None)
        use_iterable = cfg.pop('use_iterable', False)

    # Get datasets and load data
    dataset_paths = expand_dataset_paths(filepath)
    logging.info(f'Loading data from {dataset_paths}')
    dataset_list = []
    for path in dataset_paths:
        if use_iterable:
            data = MoleculeIterableDataset(filepath=path, cfg=cfg, trainer=trainer)
        else:
            data = MoleculeDataset(filepath=path, cfg=cfg, trainer=trainer)
        dataset_list.append(data)

        with open_dict(cfg):
            cfg['num_samples'] -= len(data)

        if cfg['num_samples'] < 1:
            break

    if len(dataset_list) == 1:
        dataset = dataset_list[0]
    else:
        dataset = ConcatIterableDataset(dataset_list) if use_iterable else pt_data.ConcatDataset(dataset_list)
    return dataset


def build_train_valid_test_datasets(
    cfg: DictConfig,
    trainer: Trainer
):
    # Build individual datasets.
    train_cfg = cfg.get('train_ds')
    train_dataset = _build_train_valid_test_datasets(train_cfg, trainer)

    valid_cfg = cfg.get('validation_ds', False)
    if valid_cfg:
        valid_dataset = _build_train_valid_test_datasets(train_cfg, trainer)
    else:
        valid_dataset = None

    test_cfg = cfg.get('test_ds', False)
    if test_cfg:
        test_dataset = _build_train_valid_test_datasets(test_cfg, trainer)
    else:
        test_dataset = None

    return (train_dataset, valid_dataset, test_dataset)
