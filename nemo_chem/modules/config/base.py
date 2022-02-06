# coding=utf-8

from typing import Optional, Union, Any
from dataclasses import dataclass
from nemo.core.config.base_config import Config

# Model parameters
DEFAULT_ENCODER_ARCH = 'transformer'
DEFAULT_DECODER_ARCH = 'transformer'
DEFAULT_SEQ_LEN = 512
DEFAULT_MAX_POSITION_EMBEDDINGS = 512
DEFAULT_HIDDEN_SIZE = 36 # This is tiny an intentionally set to uncommonly used sized
DEFAULT_FFN_HIDDEN_SIZE = 4 * DEFAULT_HIDDEN_SIZE
DEFAULT_NUM_LAYERS = 2
DEFAULT_ATTENTION_HEADS = 3 
DEFAULT_HIDDEN_DROPOUT = 0.1
DEFAULT_TENSOR_MODEL_PARALLEL_SIZE = 1
DEFAULT_APPLY_QUERY_KEY_LAYER_SCALING = True
DEFAULT_KV_CHANNELS = None
DEFAULT_NUM_TOKENTYPES = 0
DEFAULT_PARALLEL_OUTPUT = True
DEFAULT_PRE_PROCESS = True
DEFAULT_POST_PROCESS = True
DEFAULT_INIT_METHOD_STD = 0.02
DEFAULT_FP16_CROSS_ENTROPY = False
DEFAULT_USE_CPU_INITIALIZATION = False
DEFAULT_PRECISION = 16
DEFAULT_FP32_RESIDUAL_CONNECTION = False
DEFAULT_ACTIVATIONS_CHECKPOINT_METHOD = None
DEFAULT_ACTIVATIONS_NUM_LAYERS = 1
DEFAULT_LAYERNORM_EPSILON = 1e-5
DEFAULT_PERSIST_LAYER_NORM = False
DEFAULT_BIAS_GELU_FUSION = True
DEFAULT_OPENAI_GELU = False
DEFAULT_ONNX_SAFE = False
DEFAULT_HIDDEN_STEPS = -1
DEFAULT_HIDDEN_BLOCKS = 1
# DEFAULT_BLOCKS_MODEL = 1 # TODO are these perceiver related?
# DEFAULT_STEPS_MODEL = 16  # Latent Size = STEPS_MODEL * D_MODEL

__all__ = ["MegatronBARTModuleConfig"]

@dataclass
class MegatronBARTModuleConfig(Config):
    name: str = 'MegatronBARTModule'
    encoder_arch: str = DEFAULT_ENCODER_ARCH
    decoder_arch: str = DEFAULT_DECODER_ARCH
    vocab_size: int = 512
    max_position_embeddings: int = DEFAULT_MAX_POSITION_EMBEDDINGS
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    ffn_hidden_size: int = DEFAULT_FFN_HIDDEN_SIZE
    num_layers: int = DEFAULT_NUM_LAYERS
    num_attention_heads: int = DEFAULT_ATTENTION_HEADS
    apply_query_key_layer_scaling: bool = DEFAULT_APPLY_QUERY_KEY_LAYER_SCALING
    kv_channels: Optional[int] = DEFAULT_KV_CHANNELS
    num_tokentypes: int = DEFAULT_NUM_TOKENTYPES
    parallel_output: bool = DEFAULT_PARALLEL_OUTPUT
    pre_process: bool = DEFAULT_PRE_PROCESS
    post_process: bool = DEFAULT_POST_PROCESS
    init_method_std: float = DEFAULT_INIT_METHOD_STD
    fp16_cross_entropy: bool = DEFAULT_FP16_CROSS_ENTROPY
    use_cpu_initialization: bool = DEFAULT_USE_CPU_INITIALIZATION
    hidden_dropout: float = DEFAULT_HIDDEN_DROPOUT
    precision: int = DEFAULT_PRECISION
    fp32_residual_connection: bool = DEFAULT_FP32_RESIDUAL_CONNECTION
    activations_checkpoint_method: Optional[Any] = DEFAULT_ACTIVATIONS_CHECKPOINT_METHOD
    activations_checkpoint_num_layers: int = DEFAULT_ACTIVATIONS_NUM_LAYERS
    layernorm_epsilon: float = DEFAULT_LAYERNORM_EPSILON
    persist_layer_norm: bool = DEFAULT_PERSIST_LAYER_NORM
    bias_gelu_fusion: bool = DEFAULT_BIAS_GELU_FUSION
    openai_gelu: bool = DEFAULT_OPENAI_GELU
    onnx_safe: bool = DEFAULT_ONNX_SAFE
    hidden_steps: int = DEFAULT_HIDDEN_STEPS
    hidden_blocks: int = DEFAULT_HIDDEN_BLOCKS
    tensor_model_parallel_size: int = 1
    # pretrained: Optional[bool] = False # TODO add back after perceiver implemented
    # checkpoint_file: Optional[str] = None
    # encoder_type: str = 'seq2seq'
    # blocks_model: Optional[int] = DEFAULT_BLOCKS_MODEL
    # steps_model: Optional[int] = DEFAULT_STEPS_MODEL

    # decode_sampler: DecodeSamplerConfig = DecodeSamplerConfig()    # TODO add to model
    # optim: Optional[OptimConfig] = AdamOptimConfig()
    # train_ds: MoleculeCsvDatasetConfig = MoleculeCsvDatasetConfig()
    # validation_ds: Optional[MoleculeCsvDatasetConfig] = MoleculeCsvDatasetConfig()
    # test_ds: Optional[MoleculeCsvDatasetConfig] = MoleculeCsvDatasetConfig()