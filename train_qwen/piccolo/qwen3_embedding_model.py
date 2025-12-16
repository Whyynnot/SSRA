from __future__ import annotations

import logging
import queue
import json
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

from tqdm.autonotebook import tqdm
import numpy as np
import torch
from torch.utils.data._utils.worker import ManagerWatchdog
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers.tokenization_utils_base import BatchEncoding
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TransformersTextEmbedder(torch.nn.Module):
    def __init__(
        self,
        model: str,
        pooler_type: str = 'last',
        do_norm: bool = False,
        truncate_dim: int = 0,
        padding_left: bool = False,
        attn_type: str = 'causal',
        lora_rank: int = 0,
        adapter_path: str = '',
        quantization: bool = True,
        resume_lora: bool = False,
        freeze_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base_model = AutoModel.from_pretrained(model, quantization_config=quantization_config if quantization else None, **kwargs)
        if quantization:
            base_model = prepare_model_for_kbit_training(base_model)

        if adapter_path:
            base_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
            logger.info(f'load adapter from {adapter_path}')
            if not resume_lora:
                base_model = base_model.merge_and_unload()
                base_model.requires_grad_(True)
                logger.info(f'merge and unload adapter from {adapter_path}')

        if freeze_word_embeddings:
            base_model.embed_tokens.requires_grad_(False)

        if lora_rank:
            # LoRA config
            peft_config = LoraConfig(
                lora_alpha=lora_rank * 2,                # Scaling factor for LoRA
                lora_dropout=0.1,                        # Add slight dropout for regularization
                r=lora_rank,                             # Rank of the LoRA update matrices
                bias="none",                             # No bias reparameterization
                task_type=None,                          # Task type: CAUSAL_LM/FEATURE_EXTRACTION
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],  # Target modules for LoRA
            )

            if not adapter_path or not resume_lora:
                base_model = get_peft_model(base_model, peft_config)
                logger.info(f'add peft parameters, lora rank: {lora_rank}')

            base_model.print_trainable_parameters()

        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.tokenizer.padding_side = "left"
        self.pooler_type = pooler_type
        self.do_norm = do_norm
        self.truncate_dim = truncate_dim
        self.padding_left = padding_left
        self.attn_type = attn_type
        if pooler_type == 'first':
            assert padding_left is False
            self.pooling = self._pooling_first
        elif pooler_type == 'last':
            self.pooling = self._pooling_last
        elif pooler_type == 'mean':
            self.pooling = self._pooling_mean
        else:
            ValueError(f"Wrong pooler : {self.pooler_type}")


    def embed(
        self, 
        sentences: Sequence[str], 
        max_length: int,
        prompt: str | None = None,
        device: str | torch.device = 'cpu',
    ) -> torch.Tensor:
        inputs = self.tokenize(sentences, max_length, prompt).to(device)
        embeddings = self.forward(**inputs.data)
        return embeddings

    def tokenize(self, texts, max_length: int, prompt=None) -> BatchEncoding:
        if prompt:
            texts = [prompt + t for t in texts] 
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        return inputs

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        embeddings = self.pooling(output.last_hidden_state, attention_mask)
        if self.truncate_dim > 0:
            embeddings = embeddings[:, :self.truncate_dim]
        if self.do_norm:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask
        left_padding = (mask[:, -1].sum() == mask.shape[0])
        if left_padding:
            return hidden_state[:, -1]
        else:
            sequence_lengths = mask.sum(dim=1) - 1
            batch_size = hidden_state.shape[0]
            return hidden_state[torch.arange(batch_size, device=hidden_state.device), sequence_lengths]

    @staticmethod
    def _pooling_first(hidden_state: torch.Tensor, _) -> torch.Tensor:
        return hidden_state[:, 0]

    @staticmethod
    def _pooling_last_left(hidden_state: torch.Tensor, _) -> torch.Tensor:
        return hidden_state[:, -1]

    @staticmethod
    def _pooling_last_right(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_indices = attention_mask.sum(1) - 1
        batch_indices = torch.arange(hidden_state.size(0), device=hidden_state.device)
        pooled_output = hidden_state[batch_indices, last_indices]
        return pooled_output

    @staticmethod
    def _pooling_mean(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        assert attention_mask.ndim == 2, f"Unexpected {attention_mask.ndim=}"
        attention_mask = attention_mask.float()
        lengths = attention_mask.sum(1)
        pooled_output = torch.einsum('bsh,bs,b->bh', (hidden_state.float(), attention_mask, 1 / lengths))
        return pooled_output

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.base_model.gradient_checkpointing_enable(*args, **kwargs)


def _encode_loop(
    model: TransformersTextEmbedder,
    input_queue,
    output_queue,
    device: torch.device,
    qsize: int = 4,
    amp_dtype=None
):
    model = model.to(device)
    watchdog = ManagerWatchdog()
    keep_queue = queue.Queue(qsize + 1)

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype
        ) if amp_dtype is not None else nullcontext():
            while watchdog.is_alive():
                r = input_queue.get()
                if r is None:
                    break

                n, inputs = r
                embeddings = model.embed(*inputs, device=device)
                output_queue.put((n, embeddings))
                if keep_queue.full():
                    i = keep_queue.get()
                    del i
                keep_queue.put(embeddings)
                del r, n, inputs

    while not keep_queue.empty():
        i = keep_queue.get()
        del i
    del model, watchdog
    return
