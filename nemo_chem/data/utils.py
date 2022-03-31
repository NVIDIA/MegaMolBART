from typing import List
from enum import Enum
import re
import braceexpand
import os
from copy import deepcopy
from omegaconf import DictConfig, open_dict
import torch.utils.data as pt_data
from pytorch_lightning.trainer.trainer import Trainer

from nemo.utils import logging
from .csv_data import MoleculeDataset, MoleculeIterableDataset
from .concat import ConcatIterableDataset

__all__ = ['DatasetTypes', 'expand_dataset_paths', 'build_train_valid_test_datasets']

class DatasetTypes(Enum):
    zinc_csv  = 0


def expand_dataset_paths(filepath: str) -> List[str]:
    """Expand dataset paths from braces"""
    # TODO this should eventually be moved to a Nemo fileutils module or similar
    filepath = re.sub(r"""\(|\[|\<|_OP_""", '{', filepath) # replaces '(', '[', '<' and '_OP_' with '{'
    filepath = re.sub(r"""\)|\]|\>|_CL_""", '}', filepath) # replaces ')', ']', '>' and '_CL_' with '}'
    dataset_paths = list(braceexpand.braceexpand(filepath))
    return dataset_paths


def _build_train_valid_test_datasets(
    cfg: DictConfig, 
    trainer: Trainer,
    num_samples: int,
    filepath: str,
    metadata_path: str,
    use_iterable: bool
):

    cfg = deepcopy(cfg)
    with open_dict(cfg):
        cfg['metadata_path'] = metadata_path

    # Get datasets and load data
    dataset_paths = expand_dataset_paths(filepath)
    logging.info(f'Loading data from {dataset_paths}')
    dataset_list = []
    for path in dataset_paths:
        if use_iterable:
            data = MoleculeIterableDataset(filepath=path, cfg=cfg, trainer=trainer, num_samples=num_samples)
        else:
            data = MoleculeDataset(filepath=path, cfg=cfg, trainer=trainer, num_samples=num_samples)
        dataset_list.append(data)

        num_samples -= len(data)
        if num_samples < 1:
            break

    if len(dataset_list) == 1:
        dataset = dataset_list[0]
    else:
        dataset = ConcatIterableDataset(dataset_list) if use_iterable else pt_data.ConcatDataset(dataset_list)
    return dataset


def build_train_valid_test_datasets(
    cfg: DictConfig,
    trainer: Trainer,
    train_valid_test_num_samples: List[int]
):
    cfg = deepcopy(cfg)
    with open_dict(cfg):
        dataset_path = cfg.pop('dataset_path', '')
        dataset_files = cfg.pop('dataset_files')
        metadata_file = cfg.pop('metadata_file')
        use_iterable = cfg.pop('use_iterable', False)

    # Build individual datasets.
    filepath = os.path.join(dataset_path, 'train', dataset_files)
    metadata_path = os.path.join(dataset_path, 'train', metadata_file)
    train_dataset = _build_train_valid_test_datasets(cfg, trainer, train_valid_test_num_samples[0],
                                                     filepath, metadata_path, use_iterable)

    filepath = os.path.join(dataset_path, 'val', dataset_files)
    metadata_path = os.path.join(dataset_path, 'val', metadata_file)
    validation_dataset = _build_train_valid_test_datasets(cfg, trainer, train_valid_test_num_samples[1],
                                                          filepath, metadata_path, use_iterable)

    filepath = os.path.join(dataset_path, 'test', dataset_files)
    metadata_path = os.path.join(dataset_path, 'test', metadata_file)
    test_dataset = _build_train_valid_test_datasets(cfg, trainer, train_valid_test_num_samples[2],
                                                    filepath, metadata_path, use_iterable)

    return (train_dataset, validation_dataset, test_dataset)
