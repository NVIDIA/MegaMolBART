from typing import List
import re
import braceexpand
import numpy as np
import torch.distributed as dist
from nemo.utils import logging

__all__ = ['expand_dataset_paths']


def expand_dataset_paths(filepath: str) -> List[str]:
    """Expand dataset paths from braces"""
    # TODO this should go in a Nemo fileutils module or similar
    filepath = re.sub(r"""\(|\[|\<|_OP_""", '{', filepath) # replaces '(', '[', '<' and '_OP_' with '{'
    filepath = re.sub(r"""\)|\]|\>|_CL_""", '}', filepath) # replaces ')', ']', '>' and '_CL_' with '}'
    dataset_paths = list(braceexpand.braceexpand(filepath))
    return dataset_paths


# DEPRECATED
def shard_dataset_paths_for_ddp(self, dataset_paths):
    """Shard dataset paths for ddp"""
    dataset_paths = np.array(dataset_paths)
    num_dataset_paths = len(dataset_paths)

    # Split for data parallel
    if dist.is_initialized():
        world_size = dist.get_world_size()
        if world_size > num_dataset_paths:
            logging.warning(f'World size ({world_size}) is larger than number of data files ({num_dataset_paths}). Data will be duplicated.')
        rank = dist.get_rank()
        logging.info(f'Torch distributed is initialized with world size {world_size} and rank {rank}.')
    else:
        world_size = 1
        rank = 0
        logging.info(f'Torch distributed is not initialized.')

    num_chunks = min(num_dataset_paths, world_size)
    split_dataset_paths = np.array_split(dataset_paths, num_chunks)
    index = rank % num_chunks
    logging.info(f'Selected dataset paths {split_dataset_paths} and index {index}')
    dataset_paths = split_dataset_paths[index]          
    return dataset_paths
