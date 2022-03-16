from pathlib import Path
from nemo.utils import logging


def recursive_make_dirs(directory):
    """Recursively create required directory structure"""
    logging.info(f'Creating directory {str(directory)}...')
    if isinstance(directory, str):
        directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
