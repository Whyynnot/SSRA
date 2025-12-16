from typing import Optional, Union
from enum import Enum

from dataclasses import dataclass, field
from transformers import TrainingArguments
# from piccolo.model import PoolingStrategy


class PoolingStrategy(Enum):
    cls = 'cls'
    mean = 'mean'


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field()  # must require
    embedding_strategy: str = field(default='mean')
    extend_pe: bool = field(default=False)
    max_length: int = field(default=512)
    # scaling layer and mrl Training
    use_scaling_layer: bool = field(default=False)
    use_mrl: bool = field(default=False)
    temperature: float = field(default=0.05)
    contrast_bidirectional: bool = field(default=True)


@dataclass
class DHNMModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field()  # must require
    embedding_strategy: str = field(default='mean')
    extend_pe: bool = field(default=False)
    max_length: int = field(default=512)
    # scaling layer and mrl Training
    use_scaling_layer: bool = field(default=False)
    use_mrl: bool = field(default=False)
    temperature: float = field(default=0.07)
    contrast_bidirectional: bool = field(default=True)
    # dhnm training
    num_cached_examples: int = field(default=65536)
    num_negs_per_query: int = field(default=1)
    relative_margin: float = field(default=0.05)
    momentum: float = field(default=1.0)
    gather_examples_for_cache: bool = field(default=True)
    gather_hard_negatives: bool = field(default=False)
    gather_easy_negatives: bool = field(default=False)
    use_all_pairs: bool = field(default=True)
    enable_inf_cl: bool = field(default=False)
    enable_hard_negative_mining: bool = field(default=True)


@dataclass
class GritLMModelArguments:
    model_name_or_path: str = field()  # must require
    attn: str = field(default='bbcc')
    mode: str = field(default='embedding')
    pooling_method: str = field(default='mean')
    max_length: int = field(default=256)
    temperature: float = field(default=0.07)
    use_scaling_layer: bool = field(default=False)


@dataclass
class Qwen3ModelArguments:
    model_name_or_path: str = field()  # must require
    pooler_type: str = field(default='last')
    do_norm: bool = field(default=False)
    truncate_dim: int = field(default=0)
    padding_left: bool = field(default=True)
    attn_type: str = field(default='causal')
    # attn_implementation: str = field(default='flash_attention_2')
    lora_rank: int = field(default=0)
    adapter_path: str = field(default='')
    quantization: bool = field(default=True)
    resume_lora: bool = field(default=False)
    temperature: float = field(default=0.07)
    use_scaling_layer: bool = field(default=False)
    max_length: int = field(default=512)
    # enable_inf_cl: bool = field(default=False)
    # enable_hard_negative_mining: bool = field(default=True)
    # num_cached_examples: int = field(default=65536)
    # num_negs_per_query: int = field(default=1)
    # relative_margin: float = field(default=0.05)
    # momentum: float = field(default=1.0)

@dataclass
class Qwen3DHNModelArguments:
    model_name_or_path: str = field()  # must require
    pooler_type: str = field(default='last')
    do_norm: bool = field(default=False)
    truncate_dim: int = field(default=0)
    padding_left: bool = field(default=True)
    attn_type: str = field(default='causal')
    # attn_implementation: str = field(default='flash_attention_2')
    lora_rank: int = field(default=0)
    adapter_path: str = field(default='')
    quantization: bool = field(default=True)
    resume_lora: bool = field(default=False)
    temperature: float = field(default=0.07)
    use_scaling_layer: bool = field(default=False)
    mrl_nesting_list: list[int] = field(default_factory=lambda: [128])
    max_length: int = field(default=512)
    enable_inf_cl: bool = field(default=False)
    enable_hard_negative_mining: bool = field(default=True)
    miner_update_steps: int = field(default=0)
    num_cached_examples: int = field(default=65536)
    num_negs_per_query: int = field(default=1)
    relative_margin: float = field(default=0.05)
    momentum: float = field(default=1.0)
    cosent_weight: float = field(default=0.0)
    mask_false_negatives: bool = field(default=False)
    use_self_contrast: bool = field(default=False)
    fast_decay_temperature: bool = field(default=False)
    freeze_word_embeddings: bool = field(default=False)

@dataclass
class DataArguments:
    # train data
    # meta_paths: list[str] = field()  # must require
    root_dirs: list[str] = field()  # must require
    batch_size: int = field(default=16)
    query_prefix: str = field(default='')
    doc_prefix: str = field(default='')
    # hard neg
    neg_num: int = field(default=1)  # only affects retri_contrast_loss
    multiplex_weights: list[float] = field(default_factory=lambda: [1.0])


@dataclass
class STETrainingArguments(TrainingArguments):
    max_steps: int = field(default=-1)
    efficient_save: bool = field(default=True)
    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = field(
        default_factory=lambda: {'use_reentrant': False},
        metadata={
            "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."
        },
    )
