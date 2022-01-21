import os
import sys
import re
from setuptools import setup, find_packages
import importlib.util

package_dir = 'nemo_chem'

spec = importlib.util.spec_from_file_location('package_info', os.path.join(package_dir, 'package_info.py'))
package_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_info)
package_name = package_info.__package_name__

if os.path.exists('README.rmd'):
    with open("README.md", "r") as fh:
        long_description = fh.read()
    long_description_content_type = "text/markdown"
else:
    long_description = 'See ' + package_info.__homepage__
    long_description_content_type = 'text'

###

setup(
    name=package_name,
    version=package_info.__version__,
    description=package_info.__description__,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    url=package_info.__repository_url__,
    download_url=package_info.__download_url__,
    author=package_info.__contact_names__,
    maintainer=package_info.__contact_names__,
    license=package_info.__license__,
    packages=find_packages(include=[package_dir, package_dir + '.*']),
    include_package_data=True,
    package_dir={package_name: package_dir},
    package_data={package_name: ['vocab/*']}
)
