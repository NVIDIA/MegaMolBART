# What Is MegaMolBART v0.2?
[MegaMolBART](https://ngc.nvidia.com/catalog/models/nvidia:clara:megamolbart) is a model trained on SMILES string and this container deploys MegaMolBART model for infrencing.

# Getting Started
MegaMolBART v0.2 container encloses all prerequisites for training and inference. Please follow these steps to start **MegaMolBART** container for inference.

- Start an instance of the Docker image using the following command:

 ```
 # For Docker version 19.03 or later
 docker run \
 --gpus all \
 --rm \
 -p 8888:8888 \
 nvcr.io/nvidia/clara/megamolbart:0.2.0

 # For Docker version 19.02 or older
 docker run \
 --runtime nvidia \
 --rm \
 -p 8888:8888 \
 nvcr.io/nvidia/clara/megamolbart:0.2.0
 ```

- In a browser open URL http://<<HOSTNAME/IP>>:8888


 # License
By pulling and using the container, you accept the terms and conditions of [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0)