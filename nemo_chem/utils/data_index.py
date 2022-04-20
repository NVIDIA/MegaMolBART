import os
import time
import pickle
import multiprocessing as mp

from functools import partial

import numpy as np

from nemo.utils import logging


def _build_memmap_index_files(newline_int, fn):
    idx_fn = fn + ".idx"

    # create data map
    mdata = np.memmap(fn, dtype=np.uint8, mode='r')
    if os.path.exists(idx_fn):
        return None
    else:
        logging.info(f"Building idx file = {idx_fn}")
        midx = np.where(mdata == newline_int)[0]
        # add last item in case there is no new-line
        if (len(midx) == 0) or (midx[-1]+1 != len(mdata)):
            midx = np.asarray(midx.tolist() + [len(midx)], dtype=midx.dtype)

        size = len(mdata)
        pickle.dump(dict(midx=midx, size=size), open(idx_fn, "wb"))
        mdata._mmap.close()
        del mdata

        return True


def build_index_files(dataset_paths,
                      newline_int,
                      workers=None):
    if len(dataset_paths) < 1:
        raise ValueError("files_list must contain at leat one file name")

    if workers is None:
        workers = min(1, os.cpu_count() // 2)

    logging.info(f"Building data files")
    # load all files into memmap
    start_time = time.time()
    with mp.Pool(workers) as p:
        mdata_midx_size_list = p.map(partial(_build_memmap_index_files, newline_int),
                                     dataset_paths)
    logging.info(f'Time building mem-mapped file: {time.time() - start_time}')

    return mdata_midx_size_list