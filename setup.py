import os
import sys
import nemo
from setuptools import setup, find_packages

nemo_chem_path = nemo.__path__
nemo_chem_path = nemo_chem_path if isinstance(nemo_chem_path, str) else nemo_chem_path[0]
nemo_chem_path = os.path.join(*[nemo_chem_path, 'collections', 'chem'])
#sys.argv += ['--install-scripts', nemo_path]
#print(sys.argv)

def copy_dir(base_dir=os.getcwd(), dest_dir=nemo_chem_path):
    #dir_path =  
    #base_dir = os.path.join(nemo_chem_path, dir_path)

    for (dirpath, dirnames, files) in os.walk(base_dir):
        for f in files:
            print(f) #os.path.join(dirpath.split('/', 1)[1], f)

if __name__ == '__main__':
    copy_dir()

#setup(
#      name='nemo_chem',
#      version='0.0.1',
#      packages=find_packages(include=['.'])
#     )

