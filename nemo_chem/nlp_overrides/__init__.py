from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.utils import AppState, logging
from torch.nn.parallel import DistributedDataParallel
from pytorch_lightning.overrides import LightningDistributedModule

# Temporary solution to issue with find_unused_parameters
class MegaMolBARTNLPDDPPlugin(NLPDDPPlugin):
    def configure_ddp(self):
        """ Override LightningModule ddp if using model parallel.
            Sets find_unused_parameters to False to use activation-checkpoint-recomputation.
        """

        app_state = AppState()

        if app_state.model_parallel_size is not None:
            logging.info(f"Configuring DDP for model parallelism.")

            # With model parallelism, multiple GPUs form a large "logical GPU"
            # this means that data parallel groups span multiple GPUs
            # and are non-trivial
            device_ids = self.determine_ddp_device_ids()
            self._model = DistributedDataParallel(
                LightningDistributedModule(self.model),
                device_ids=device_ids,
                output_device=device_ids[0],
                process_group=app_state.data_parallel_group,
                find_unused_parameters=True,
                **self._ddp_kwargs,
            )

        else:
            super().configure_ddp()