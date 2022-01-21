import os
import sys
import re
import nemo
from setuptools import setup, find_packages
import importlib.util

nemo_chem_path = os.path.join(*['nemo', 'collections', 'chem'])
# sys.argv += ['--install-scripts', nemo_path]
# print(sys.argv)

spec = importlib.util.spec_from_file_location('package_info', 'nemo/package_info.py')
package_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_info)

__package_name__ = package_info.__package_name__
__contact_names__ = package_info.__contact_names__
__homepage__ = package_info.__homepage__
__repository_url__ = package_info.__repository_url__
__download_url__ = package_info.__download_url__
__description__ = package_info.__description__
__license__ = package_info.__license__
__keywords__ = package_info.__keywords__
__shortversion__ = package_info.__shortversion__
__version__ = package_info.__version__

if os.path.exists('README.rmd'):
    with open("README.md", "r") as fh:
        long_description = fh.read()
    long_description_content_type = "text/markdown"
else:
    long_description = 'See ' + __homepage__

###

def copy_dir(dest_dir=nemo_chem_path):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, dest_dir)

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
    name=__package_name__,
    version=__version__,
    description=__description__,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    url=__repository_url__,
    download_url=__download_url__,
    author=__contact_names__,
    maintainer=__contact_names__,
    license=__license__,
    packages=find_packages(include=[nemo_chem_path]),
    data_files=[copy_dir()]
)