import logging
import torch

from typing import List

from omegaconf import OmegaConf

from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.parts.nlp_overrides import (NLPDDPPlugin,
                                                      NLPSaveRestoreConnector)
from nemo.utils.app_state import AppState
from nemo_chem.data import MoleculeEnumeration
from nemo_chem.models.megamolbart import MegaMolBARTModel


log = logging.getLogger(__name__)
__all__ = ["NeMoMegaMolBARTWrapper"]


class NeMoMegaMolBARTWrapper():
    '''
    Implements functions to infer using MegaMolBART model
    '''

    def __init__(self,
                 model_cfg=None,
                 random_weights=False) -> None:
        super().__init__()

        if model_cfg is None:
            # TODO: Create a default global variable for this
            log.info('Loading default configuration...')
            model_cfg = OmegaConf.load(
                '/workspace/nemo_chem/examples/chem/conf/infer.yaml')
        if random_weights:
            model_cfg['model']['model_path'] = None

        self.model = self.load_model(model_cfg)
        self.cfg = self.model._cfg
        self.max_seq_len = self.cfg.max_position_embeddings
        self.tokenizer = self.model.tokenizer

        pad_size_divisible_by_8 = True if self.cfg.masked_softmax_fusion else False
        self.mol_enum = MoleculeEnumeration(tokenizer=self.tokenizer,
                                            seq_length=self.cfg.seq_length,
                                            pad_size_divisible_by_8=pad_size_divisible_by_8,
                                            **self.cfg.data)
        self.mol_enum.encoder_mask = False

        self.mol_enum.encoder_augment = False
        self.mol_enum.encoder_mask = False
        self.mol_enum.canonicalize_input = False
        self.mol_enum.decoder_augment = False
        self.mol_enum.decoder_mask = False
        self.mol_enum.mask_prob = 0

    def _tokenize(self, smis: List[str]):
        tokens = [self.tokenizer.text_to_tokens(s) for s in smis]
        token_ids = [self.tokenizer.token_to_ids(t) for t in tokens]

        pad_length = max([len(seq) for seq in token_ids])
        # if self.pad_size_divisible_by_8:
        #     pad_length = int(math.ceil(pad_length/8) * 8)

        # 1/True = Active, 0/False = Inactive
        encoder_mask = [([1] * len(seq)) + ([0] * (pad_length - len(seq))) for seq in token_ids]
        token_ids = [seq + ([self.tokenizer.pad_id] * (pad_length - len(seq))) for seq in token_ids]

        token_ids = torch.tensor(token_ids, dtype=torch.int64).cuda()
        encoder_mask = torch.tensor(encoder_mask,
                                    dtype=torch.int64,
                                    device=token_ids.device)

        return token_ids, encoder_mask

    def _transform(self, smis):
        '''
        Transforms SMILES into hidden state.

        Args:
            smis (list[str]): list of SMILES strings

        Returns:
            tokens_enc (torch.Tensor, long): token ID values for samples
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections
        '''

        tokens_enc, enc_mask = self._tokenize(smis)
        hidden_states = self.model.encode(tokens_enc, enc_mask)

        return hidden_states, enc_mask

    def load_model(self, model_cfg):
        """Load saved model checkpoint

        Params:
            checkpoint_path: path to nemo checkpoint

        Returns:
            MegaMolBART trained model
        # """
        torch.set_grad_enabled(False)

        # trainer required for restoring model parallel models
        trainer = Trainer(
            plugins=NLPDDPPlugin(),
            devices=1,
            accelerator='gpu',
            precision=32,
        )

        app_state = AppState()
        if model_cfg.model.model_path is not None:
            model = MegaMolBARTModel.restore_from(
                restore_path=model_cfg.model.model_path,
                trainer=trainer,
                save_restore_connector=NLPSaveRestoreConnector(),
            )
        else:
            # Initialize with random weights
            cfg = OmegaConf.load(
                '/workspace/nemo_chem/examples/chem/conf/megamolbart_pretrain_base.yaml')
            cfg.model.num_layers=6
            cfg.model.hidden_size=512
            cfg.model.num_attention_heads=8
            cfg.model.precision = cfg.trainer.precision

            model = MegaMolBARTModel(cfg.model, trainer)

        model.freeze()

        return model

    def smis_to_hidden(self, smis: List[str]):
        """Compute hidden-state and padding mask for smiles.

        Params
            smi: string, input SMILES molecule

        Returns
            hidden-state array and boolean mask
        """
        if isinstance(smis, str):
            smis = [smis]

        hidden_states, enc_masks = self._transform(smis)
        return hidden_states, enc_masks

    def smis_to_embedding(self, smis: List[str]):
        """Computes embedding and padding mask for smiles.

        Params
            smi: string, input SMILES molecule

        Returns
            hidden-state array and boolean mask
        """
        if isinstance(smis, str):
            smis = [smis]

        hiddens, _ = self.smis_to_hidden(smis)
        return torch.mean(hiddens, dim=1)

    def hidden_to_smis(self, hidden_states, enc_mask):

        predicted_tokens_ids, _ = self.model.decode(None,
                                                    enc_mask,
                                                    self.cfg.max_position_embeddings,
                                                    enc_output=hidden_states)

        predicted_tokens_ids = predicted_tokens_ids.cpu().detach().numpy().tolist()
        for i, predicted_token_id in enumerate(predicted_tokens_ids):
            if self.tokenizer.eos_id in predicted_token_id:
                idx = predicted_token_id.index(self.tokenizer.eos_id)
                predicted_tokens_ids[i] = predicted_token_id[:idx]
            else:
                predicted_tokens_ids[i] = [id for id in predicted_token_id if id != self.tokenizer.pad_id]

        smis = self.tokenizer.ids_to_text(predicted_tokens_ids)

        return smis

    def sample(self,
               smis,
               num_samples=10,
               return_embedding=False,
               sampling_method='greedy-perturbate',
               sampling_kwarg={'scaled_radius': 1, 'topk': 10}):
        """
        Sample from model given hidden states and mask
        """
        hidden_states, enc_masks = self.smis_to_hidden(smis)

        if sampling_method == 'greedy-perturbate':
            scaled_radius = sampling_kwarg['scaled_radius']
            sample_masks = enc_masks.repeat_interleave(num_samples, 0)
            perturbed_hiddens = hidden_states.repeat_interleave(num_samples, 0)
            perturbed_hiddens = perturbed_hiddens + (scaled_radius * torch.randn(perturbed_hiddens.shape).to(perturbed_hiddens.device))

            samples = self.hidden_to_smis(perturbed_hiddens, sample_masks)
            if return_embedding:
                embs = torch.mean(perturbed_hiddens, dim=1)
        else:
            raise ValueError(f'Invalid samping method {sampling_method}')

        if return_embedding:
            return samples, embs
        else:
            return samples
