import os
from dataclasses import dataclass, field
import torch
from transformers import AutoConfig


@dataclass
class CosyVoice2LLMConfig:
    architectures: list[str] = field(default_factory=lambda: ["Qwen2ForCausalLM"])
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 6561  # speech eos
    hidden_act: str = "silu"
    hidden_size: int = 896
    initializer_range: float = 0.02
    intermediate_size: int = 4864
    max_position_embeddings: int = 32768
    max_window_layers: int = 24
    model_type: str = "qwen2"
    num_attention_heads: int = 14
    num_hidden_layers: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 64
    rms_norm_eps: float = 1e-06
    rope_scaling: dict | None = None
    rope_theta: float = 1000000.0
    sliding_window: int = 32768
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    transformers_version: str = "4.52.0.dev0"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 158500  # text_vocab_size + speech_vocab_size + 2 (eos and task_id)
    text_vocab_size: int = 151936
    speech_vocab_size: int = 6562  # actually 6564, we only care about non-streaming inference, so cut off tokens (6562, 6563) that are only used for streaming TTS
    lm_head_bias: bool = True
    qkv_bias: bool = True


@dataclass
class SamplingParams:
    temperature: float = 1.0
    min_tokens: int = 2
    max_tokens: int = 64
    ignore_eos: bool = False
    top_k: int = 25


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: CosyVoice2LLMConfig | AutoConfig = field(default_factory=CosyVoice2LLMConfig)
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    sampling_params: SamplingParams = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.sampling_params = SamplingParams() if self.sampling_params is None else self.sampling_params

        max_pos = getattr(self.hf_config, "max_position_embeddings", 4096)
        self.max_model_len = min(self.max_model_len, max_pos)
        assert self.max_num_batched_tokens >= self.max_model_len
