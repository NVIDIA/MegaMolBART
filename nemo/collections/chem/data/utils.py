from typing import List
import re
import braceexpand
import numpy as np
import torch.distributed as dist
from nemo.utils import logging

__all__ = ['expand_dataset_paths', 'check_seq_len']


def expand_dataset_paths(filepath: str) -> List[str]:
    """Expand dataset paths from braces"""
    # TODO this should go in a core-level NeMo fileutils module or similar
    open_parenthesis_regex = re.compile(r"""\(|\[|\<|_OP_""")
    close_parenthesis_regex = re.compile(r"""\)|\]|\>|_CL_""")
    filepath = re.sub(open_parenthesis_regex, '{', filepath) # replaces '(', '[', '<' and '_OP_' with '{'
    filepath = re.sub(close_parenthesis_regex, '}', filepath) # replaces ')', ']', '>' and '_CL_' with '}'
    dataset_paths = list(braceexpand.braceexpand(filepath))
    return dataset_paths


def check_seq_len(tokens: List[List[str]], mask: List[List[int]], max_seq_len: int):
    """ Warn user and shorten sequence if the tokens are too long, otherwise return original

    Args:
        tokens (List[List[str]]): List of token sequences
        mask (List[List[int]]): List of mask sequences

    Returns:
        tokens (List[List[str]]): List of token sequences (shortened, if necessary)
        mask (List[List[int]]): List of mask sequences (shortened, if necessary)
    """

    seq_len = max([len(ts) for ts in tokens])
    if seq_len > max_seq_len:
        logging.warning(f'Truncating sequences of length {seq_len} which is longer than maximum ({max_seq_len}).')
        tokens_short = [ts[:max_seq_len] for ts in tokens]
        mask_short = [ms[:max_seq_len] for ms in mask]
        return (tokens_short, mask_short)
    return (tokens, mask)


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
