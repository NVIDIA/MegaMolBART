# Model Overview

MegaMolBART is a model that understands chemistry and can be used for a variety of cheminformatics applications in drug discovery. The embeddings from its encoder can be used as features for predictive models. Alternatively, the encoder and decoder can be used together to generate novel molecules by sampling the model's embedding space.

## Model Architecture

The model is a seq2seq transformer called a Bidirectional and Auto-Regressive Transformer (BART) [1]. It is based on a previous model, called Chemformer [2]. Pre-norm layer normalization and GELU activation are used throughout. This version of MegaMolBART has 8 layers, 4 attention heads, and a hidden space dimension of 256. Dropout was set to 0.1 during training.

## Training

MegaMolBART was developed using the Nemo-Megatron framework. It was trained with data parallelism on 64 V100 GPUs (4 nodes x 16 GPUs) for 8 epochs (approximately 160k iterations or ~80 wall-clock hours) using a batch size of 32 molecules per GPU (micro batch). The Noam scheduler was used, with a peak learning rate value of 0.0005 and ~8000 warmup steps. FusedAdam optimization was used with parameters beta1 = 0.9 and beta2 = 0.999. Categorical cross-entropy loss was used to train the model.

## Dataset

The ZINC-15 database was used for training [3]. Approximately 1.45 Billion molecules (SMILES strings) were selected from [tranches](https://zinc15.docking.org/tranches/home/) meeting the following constraints: molecular weight <= 500 Daltons, LogP <= 5, reactivity level was "reactive", and purchasability was "annotated". The compounds were filtered to ensure a maximum length of 512 characters. Train, validation, and test splits were randomly split as 99% / 0.5% / 0.5%.

Data augmentation during training was performed via masking and SMILES randomization as described previously [2].


## How to Use this Model
MegaMolBART can be run on hardware with access to any NVIDIA GPU with memory greater than 8 GB. The model can also be used for:

- Fetching embeddings from the latent space representing a SMILES string
- SMILES strings sampled around the region of a input SMILES molecule


## References

1. Lewis M., Liu Y., Goyal N., Ghazvininejad M., Mohamed A., Levy O., Stoyanov V., and Zettlemoyer L., "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension", *ArXiv*, 2019, [doi](https://arxiv.org/abs/1910.13461).
2. Irwin R., Dimitriadis S., He J., and Bjerrum E., "Chemformer: A Pre-Trained Transformer for Computational Chemistry", *Mach. Learn.: Sci. Technol.*, 2022, [doi](https://iopscience.iop.org/article/10.1088/2632-2153/ac3ffb).
3. Sterling T., and Irwin, J., *Chem. Inf. Model*, 2015, [doi](http://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559).

# License

## Apache License 2.0

A permissive license whose main conditions require preservation of copyright and license notices. Contributors provide an express grant of patent rights. Licensed works, modifications, and larger works may be distributed under different terms and without source code. The full copy of the license is available in [the repository](../../LICENSE).
