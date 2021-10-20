from dataclasses import dataclass
from typing import Tuple, Optional, Any
from torch.optim.lr_scheduler import LambdaLR
from dataclasses import dataclass
from nemo.core.config import SchedulerParams
from nemo.core.config.modelPT import OptimConfig


__all__ = ["AdamOptimConfig", "TransformerLR", "TransformerLRParams"]

@dataclass
class AdamOptimConfig(OptimConfig):
    lr: float = 1.0 # TODO this is what was in the paper, I believe it is wrong
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    sched: Optional[Any]


@dataclass
class TransformerLRParams(SchedulerParams):
    lr: float = AdamOptimConfig.lr
    warm_up_steps: int = 0
    d_model: int = 256 # TODO how to automatically set to MegatronBARTConfig.d_model
    verbose: bool = False


class TransformerLR(LambdaLR):
    def __init__(self, optimizer, lr, warm_up_steps, d_model, 
                 last_epoch: int = -1, verbose: bool = False, **kwargs):
        self.lr = lr
        self.warm_up_steps = warm_up_steps
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