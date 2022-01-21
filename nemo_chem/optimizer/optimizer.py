from dataclasses import dataclass
from typing import Tuple, Optional, Any, Union
from torch.optim.lr_scheduler import LambdaLR
from dataclasses import dataclass
from nemo.core.config import SchedulerParams
from nemo.core.config.modelPT import OptimConfig
from apex.transformer import parallel_state


__all__ = ["AdamOptimConfig", "TransformerLR", "TransformerLRParams"]

@dataclass
class AdamOptimConfig(OptimConfig):
    lr: float = 1.0 # TODO check this for finetuning
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    sched: Optional[Any]


@dataclass
class TransformerLRParams(SchedulerParams):
    lr: float = 1.0
    warm_up_steps: Optional[int] = None
    d_model: int = 256 # TODO how to automatically set to MegatronBARTConfig.d_model
    verbose: bool = False
    micro_batch_size: int = 1


class TransformerLR(LambdaLR):
    def __init__(self, optimizer, lr, warm_up_steps, d_model, 
                 last_epoch: int = -1, verbose: bool = False, micro_batch_size: int = 1, **kwargs):
        chemformer_global_batch_size = 4 * 128 # data_parallel_size * micro_batch_size
        scale_factor = chemformer_global_batch_size / (parallel_state.get_data_parallel_world_size() * micro_batch_size)

        self.lr = lr
        if warm_up_steps:
            self.warm_up_steps = warm_up_steps
        else:
            self.warm_up_steps = int( 8000 * scale_factor ) # Scale based on Chemformer paper

        self.d_model = d_model
        super(TransformerLR, self).__init__(optimizer, 
                                     lr_lambda=self._transformer_lr_lambda, 
                                     last_epoch=last_epoch, verbose=verbose)

    def _transformer_lr_lambda(self, step):
        mult = self.d_model ** -0.5
        step = 1 if step == 0 else step  # Stop div by zero errors
        lr = min(step ** -0.5, step * (self.warm_up_steps ** -1.5))
        return self.lr * mult * lr

    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]