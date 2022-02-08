from dataclasses import dataclass
from typing import Tuple, Optional, Any
from dataclasses import dataclass
from nemo.core.config.modelPT import OptimConfig


__all__ = ["AdamOptimConfig"]

# TODO Change to nemo.core.config.optimizers.AdamParams and remove this file if Config is needed
@dataclass
class AdamOptimConfig(OptimConfig):
    lr: float = 1.0 # TODO check this for finetuning
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    sched: Optional[Any]
