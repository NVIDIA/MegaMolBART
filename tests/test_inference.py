import logging
from contextlib import contextmanager
from rdkit import Chem

from hydra import compose, initialize
from nemo_chem.models.megamolbart import NeMoMegaMolBARTWrapper

log = logging.getLogger(__name__)

_INFERER = None


@contextmanager
def load_model(inf_cfg):

    global _INFERER
    if _INFERER is None:
        _INFERER = NeMoMegaMolBARTWrapper(model_cfg=inf_cfg)
    yield _INFERER


def test_smis_to_hiddens():
    with initialize(config_path="../examples/chem/conf"):
        cfg = compose(config_name="infer")

        with load_model(cfg) as inferer:
            smis = ['c1cc2ccccc2cc1',
                    'COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC']
            hidden_state, pad_masks = inferer.smis_to_hidden(smis)

            assert hidden_state is not None
            assert hidden_state.shape[0] == len(smis)
            assert hidden_state.shape[2] == inferer.cfg.max_position_embeddings
            assert pad_masks is not None


def test_smis_to_embedding():
    with initialize(config_path="../examples/chem/conf"):
        cfg = compose(config_name="infer")

        with load_model(cfg) as inferer:
            smis = ['c1cc2ccccc2cc1',
                    'COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC']
            embedding = inferer.smis_to_embedding(smis)

            assert embedding is not None
            assert embedding.shape[0] == len(smis)
            assert embedding.shape[1] == inferer.cfg.max_position_embeddings


def test_hidden_to_smis():
    with initialize(config_path="../examples/chem/conf"):
        cfg = compose(config_name="infer")

        with load_model(cfg) as inferer:
            smis = ['c1cc2ccccc2cc1',
                    'COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC']
            hidden_state, pad_masks = inferer.smis_to_hidden(smis)
            infered_smis = inferer.hidden_to_smis(hidden_state, pad_masks)
            log.info(f'Input SMILES and Infered: {smis}, {infered_smis}')

            assert(len(infered_smis) == len(smis))

            for smi, infered_smi in zip(smis, infered_smis):
                log.info(f'Input SMILES:{smi},  {infered_smi}')
                input_mol = Chem.MolFromSmiles(smi)
                infer_mol = Chem.MolFromSmiles(infered_smi)
                assert input_mol is not None and infer_mol is not None

                canonical_smi = Chem.MolToSmiles(input_mol, canonical=True)
                canonical_infered_smi = Chem.MolToSmiles(infer_mol, canonical=True)
                log.info(f'Canonical Input and Infered: {canonical_smi}, {canonical_infered_smi}')

                assert(canonical_smi == canonical_infered_smi)


def test_sample():
    with initialize(config_path="../examples/chem/conf"):
        cfg = compose(config_name="infer")

        with load_model(cfg) as inferer:
            smis = ['c1cc2ccccc2cc1',
                    'COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC']
            samples = inferer.sample(smis, num_samples=10, sampling_method='greedy-perturbate')
            log.info(smis)
            log.info(samples)