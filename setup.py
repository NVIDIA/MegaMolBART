import os
import sys
import re
import nemo
from setuptools import setup, find_packages

nemo_chem_path = os.path.join(*['nemo', 'collections', 'chem'])
# sys.argv += ['--install-scripts', nemo_path]
# print(sys.argv)

def copy_dir(dest_dir=nemo_chem_path):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, 'nemo', 'collections', 'chem')

    pattern_list = [r"""^\.""", r"""\.code-workspace$""", r"""\.pyc$""", r"""__pycache__"""]
    filter_func = lambda pattern: any([re.search(x[0], x[1]) is not None 
                                    for x in [(pattern, rel_path), (pattern, f)]])
    data_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), base_dir)            
            filter_file = any(list(map(filter_func, pattern_list)))
            if (not filter_file) & (f != '__init__.py'):
                data_files.append(rel_path)
    return (dest_dir, data_files)

setup(
     name='nemo_chem',
     version='0.0.1',
     packages=find_packages(include=['nemo/collections/chem']),
     data_files=[copy_dir()]
    )
