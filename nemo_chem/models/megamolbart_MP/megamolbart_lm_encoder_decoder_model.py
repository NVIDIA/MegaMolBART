import os
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_lm_encoder_decoder_model import MegatronLMEncoderDecoderModel

from nemo_chem.tokenizer import MolEncTokenizer, DEFAULT_VOCAB_PATH
from nemo_chem.data import MoleculeEnumeration

__all__ = ["MegaMolBARTLMEncoderDecoderModel"]


class MegaMolBARTLMEncoderDecoderModel(MegatronLMEncoderDecoderModel):
    """

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self._tokenizer_config = cfg.tokenizer  # TODO remove this with get_cheminformatics_tokenizer
        super().__init__(cfg.model, trainer=trainer)

    def _build_tokenizer(self):
        """
        Tokenizer from MegaMolBART.
        """
        vocab_path = self._tokenizer_config.get('vocab_path', DEFAULT_VOCAB_PATH) # TODO remove this with get_cheminformatics_tokenizer
        if not os.path.exists(vocab_path):
            raise ValueError(f'Vocab file not found at {vocab_path}')

        self.tokenizer = MolEncTokenizer.from_vocab_file(vocab_path=vocab_path, **self._tokenizer_config)

    def _build_vocab(self):
        """
        Manipulate vocabulary (e.g., pad vocabulary for increased performance)/
        """
        # TODO: add config to allow to disable it?
        self.padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=len(self.tokenizer),
            make_vocab_size_divisible_by=self._cfg.get('make_vocab_size_divisible_by', 128),
            tensor_model_parallel_size=self._cfg.get('tensor_model_parallel_size', 1),
        )

    # def training_step():

    # def validation_step():

    # def validation_epoch_end():

    # def test_step():

    # def test_epoch_end()

    # def loss_func():

    # def process_batch():

    # def predict_step():

    # def decode():

    # def compelte --> decode
    