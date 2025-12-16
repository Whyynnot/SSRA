# Some code in this file is referenced from: moco
# Project Name: facebookresearch/moco
# Author: https://github.com/facebookresearch
# Repository Address: https://github.com/facebookresearch/moco
# License Agreement: CC-BY-NC 4.0
# Original File: https://github.com/facebookresearch/moco/blob/main/moco/builder.py#L12



import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import os
import logging

from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import ClassVar, Literal, Type, TypeVar, Union, cast
from dataclasses import dataclass
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel  # type: ignore
from transformers.modeling_outputs import ModelOutput

from piccolo.qwen3_embedding_model import TransformersTextEmbedder
from inf_cl import cal_inf_loss
from utils.dist_utils import all_gather_with_grad, concat_all_gather


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# @dataclass
# class DROutput(ModelOutput):
#     q_reps: torch.Tensor = None
#     p_reps: torch.Tensor = None
#     logit_scale_exp: torch.Tensor = None


def create_attention_mask_from_input_ids(input_ids: torch.Tensor, pad_token_id: int, padding_left: bool = True) -> torch.Tensor:
    attention_mask = (input_ids != pad_token_id).to(input_ids.dtype)
    if padding_left:
        attention_mask[..., -1] = 1
    return attention_mask


class ScalingLayer(torch.nn.Module):
    def __init__(self, origin_dim: int = 1024, scaling_dim: int = 1792):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=origin_dim, out_features=scaling_dim, bias=True)

    def forward(self, input):
        return self.linear(input)


class Qwen3Embedder(TransformersTextEmbedder):
    def __init__(
        self,
        model_name_or_path: str,
        pooler_type: str = 'last',
        do_norm: bool = False,
        truncate_dim: int = 0,
        padding_left: bool = False,
        attn_type: str = 'causal',
        lora_rank: int = 0,
        adapter_path: str = '',
        quantization: bool = True,
        resume_lora: bool = False,
        # use_gradient_checkpointing: bool = True,
        temperature: float = 0.07,
        use_scaling_layer: bool = False,
        mrl_nesting_list: list = [128, 512, 768, 1024],
        enable_inf_cl: bool = False,
        cosent_weight: float = 0.0,
        mask_false_negatives: bool = False,
        use_self_contrast: bool = False,
        fast_decay_temperature: bool = False,
        freeze_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=model_name_or_path,
            pooler_type=pooler_type,
            do_norm=do_norm,
            truncate_dim=truncate_dim,
            padding_left=padding_left,
            attn_type=attn_type,
            lora_rank=lora_rank,
            adapter_path=adapter_path,
            quantization=quantization,
            resume_lora=resume_lora,
            freeze_word_embeddings=freeze_word_embeddings,
            **kwargs,
        )

        self.temperature_bias = torch.nn.Parameter(torch.ones([]) * temperature)
        self.logit_scale_bias = torch.nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.cosent_temperature_bias = torch.nn.Parameter(torch.ones([]) * 0.1)
        self.cosent_logit_scale_bias = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.1))
        self.mrl_nesting_list = mrl_nesting_list
        self.enable_inf_cl = enable_inf_cl
        self.cosent_weight = cosent_weight
        self.mask_false_negatives = mask_false_negatives
        self.use_self_contrast = use_self_contrast
        self.fast_decay_temperature = fast_decay_temperature

        self.use_scaling_layer = use_scaling_layer
        if use_scaling_layer:
            '''
            Here we hard code the scaling layer pretrain path, input_dim and output_dim, you can modify it by yourself
            '''
            self.scaling_layer = ScalingLayer(origin_dim=1024, scaling_dim=2048)
            if os.path.exists(os.path.join(model_name_or_path, '2_Dense/pytorch_model.bin')):
                scaling_layer_state_dict = torch.load(os.path.join(model_name_or_path, '2_Dense/pytorch_model.bin'))
                self.scaling_layer.load_state_dict(scaling_layer_state_dict, strict=True)
                print('load scaling layer successfully')
            elif os.path.exists(os.path.join(adapter_path, '2_Dense/pytorch_model.bin')):
                scaling_layer_state_dict = torch.load(os.path.join(adapter_path, '2_Dense/pytorch_model.bin'))
                self.scaling_layer.load_state_dict(scaling_layer_state_dict, strict=True)
                print('load scaling layer successfully')
            else:
                print('not found pretrain, random init scaling layer')

        self.metric_hook = dict(accuracy=[], temperature=[], loss_q2i=[])
        if cosent_weight:
            self.metric_hook['loss_cosent'] = []
            self.metric_hook['cosent_temperature'] = []

    def get_query_passage_reps(self, query, passage, **kwargs):
        q_reps = self.get_embedding(query, **kwargs).float()
        p_reps = self.get_embedding(passage, **kwargs).float()
        return q_reps, p_reps

    def get_embedding(self, input_ids: torch.LongTensor, attention_mask: Union[torch.Tensor, None] = None, **kwargs):
        if input_ids is None:
            return None
        if attention_mask is None:
            attention_mask = create_attention_mask_from_input_ids(input_ids, self.pad_token_id, self.padding_left)
        emb = super().forward(input_ids, attention_mask, **kwargs)
        if self.use_scaling_layer:
            emb = self.scaling_layer(emb)
        return emb.float()

    def contrastive_loss_fn(self, logit_scale_exp, anchor, pos, neg=None):
        if not self.enable_inf_cl:
            rank, bs = dist.get_rank(), anchor.size(0)
            targets = torch.linspace(rank * bs, (rank + 1) * bs - 1, bs, dtype=torch.int32, device=anchor.device).long()
            sim_q2t = anchor @ pos.t()
            if neg is not None:
                sim_neg = anchor @ neg.t()
                sim_q2t = torch.cat([sim_q2t, sim_neg], dim=1)

            if self.mask_false_negatives:
                with torch.no_grad():
                    thresholds = sim_q2t[torch.arange(sim_q2t.size(0)), targets].view(-1, 1) + 0.1
                    mask = sim_q2t > thresholds
                    sim_q2t.masked_fill_(mask, -1e4)

            sim_q2t *= logit_scale_exp
            loss_spec = F.cross_entropy(sim_q2t, targets, reduction='none')
        else:
            candidate = torch.cat([pos, neg], dim=0) if neg is not None else pos
            loss_spec = cal_inf_loss(anchor, candidate, scale=logit_scale_exp, head_dim=min(anchor.size(1), 256))
        return loss_spec

    def contrastive_loss_with_self_contrast(self, logit_scale_exp, anchor, pos, all_anchor=None, all_pos=None, neg=None):
        rank, bs = dist.get_rank(), anchor.size(0)
        targets = torch.linspace(rank * bs, (rank + 1) * bs - 1, bs, dtype=torch.int32, device=anchor.device).long()
        sim_q2t = anchor @ all_pos.t()
        sim_q2q = anchor @ all_anchor.t()
        sim_t2t = pos @ all_pos.t()
        sim_q2q[torch.arange(sim_q2q.size(0)), targets] = -1e4
        sim_t2t[torch.arange(sim_t2t.size(0)), targets] = -1e4
        sim_q2t = torch.cat([sim_q2t, sim_q2q, sim_t2t], dim=1)
        if neg is not None:
            sim_neg = anchor @ neg.t()
            sim_q2t = torch.cat([sim_q2t, sim_neg], dim=1)

        if self.mask_false_negatives:
            thresholds = sim_q2t[torch.arange(sim_q2t.size(0)), targets].view(-1, 1) + 0.1
            thresholds = thresholds.detach()
            mask = sim_q2t > thresholds
            sim_q2t.masked_fill_(mask, -1e4)

        sim_q2t *= logit_scale_exp
        loss_spec = F.cross_entropy(sim_q2t, targets, reduction='none')
        return loss_spec

    def cosent_loss_fn(self, logit_scale_exp, anchor, pos, label):
        predict_similarity = torch.cosine_similarity(anchor, pos, dim=-1)
        cosine_similarity_diff = -(predict_similarity.unsqueeze(0) - predict_similarity.unsqueeze(1))
        smaller_mask = label.unsqueeze(0) <= label.unsqueeze(1)
        cosine_similarity_diff = cosine_similarity_diff[~smaller_mask]

        if self.mask_false_negatives:
            with torch.no_grad():
                mask = cosine_similarity_diff > 0.1
                cosine_similarity_diff.masked_fill_(mask, -1e4)

        cosine_similarity_diff = cosine_similarity_diff * logit_scale_exp
        bias = torch.tensor([0.0], dtype=cosine_similarity_diff.dtype, device=cosine_similarity_diff.device)
        cosine_diff_scores_add_bias = torch.cat((cosine_similarity_diff, bias))
        loss = torch.logsumexp(cosine_diff_scores_add_bias, dim=0)
        return loss

    def compute_contrastive_loss(
        self,
        text_ids: torch.Tensor,
        text_pos_ids: torch.Tensor,
        text_neg_ids: Union[torch.Tensor, None] = None,
        query_ids: Union[torch.Tensor, None] = None,
        labels: Union[torch.Tensor, None] = None,
        **kwargs,
    ):
        text_embeddings = self.get_embedding(text_ids, **kwargs)
        text_pos_embeddings = self.get_embedding(text_pos_ids, **kwargs)
        text_neg_embeddings = self.get_embedding(text_neg_ids, **kwargs)
        query_embeddings = self.get_embedding(query_ids, **kwargs)

        if self.fast_decay_temperature:
            logit_scale_exp = 1.0 / self.temperature_bias
            cosent_logit_scale_exp = 1.0 / self.cosent_temperature_bias
        else:
            logit_scale_exp = self.logit_scale_bias.exp()
            cosent_logit_scale_exp = self.cosent_logit_scale_bias.exp()

        loss = torch.tensor(0.0, device=text_embeddings.device)
        loss_cosent = torch.tensor(0.0, device=text_embeddings.device)
        loss_q2i = torch.tensor(0.0, device=text_embeddings.device)

        if self.cosent_weight and labels is not None:
            for num_feat in self.mrl_nesting_list:
                emb, pos_emb = text_embeddings[..., :num_feat], text_pos_embeddings[..., :num_feat]
                emb, pos_emb = F.normalize(emb, dim=-1), F.normalize(pos_emb, dim=-1)
                loss_spec = self.cosent_loss_fn(cosent_logit_scale_exp, emb, pos_emb, labels)
                loss_cosent += loss_spec / len(self.mrl_nesting_list)
            loss += self.cosent_weight * loss_cosent

        if not self.enable_inf_cl or self.use_self_contrast:
            all_text_embeddings = all_gather_with_grad(text_embeddings)
            all_text_pos_embeddings = all_gather_with_grad(text_pos_embeddings)
            all_query_embeddings = all_gather_with_grad(query_embeddings) if query_embeddings is not None else None

        for num_feat in self.mrl_nesting_list:
            emb, pos_emb = text_embeddings[..., :num_feat], text_pos_embeddings[..., :num_feat]
            emb, pos_emb = F.normalize(emb, dim=-1), F.normalize(pos_emb, dim=-1)
            all_emb, all_pos_emb = all_text_embeddings[..., :num_feat], all_text_pos_embeddings[..., :num_feat]
            all_emb, all_pos_emb = F.normalize(all_emb, dim=-1), F.normalize(all_pos_emb, dim=-1)
            neg_emb = None
            if text_neg_embeddings is not None:
                neg_emb = text_neg_embeddings[..., :num_feat]
                neg_emb = F.normalize(neg_emb, dim=-1)

            if query_embeddings is not None:
                q_emb = F.normalize(query_embeddings[..., :num_feat], dim=-1)
                all_q_emb = F.normalize(all_query_embeddings[..., :num_feat], dim=-1)

            if self.use_self_contrast:
                loss_spec = self.contrastive_loss_with_self_contrast(logit_scale_exp, emb, pos_emb, all_emb, all_pos_emb, neg_emb)
                if query_embeddings is not None:
                    loss_spec_q2i = self.contrastive_loss_with_self_contrast(logit_scale_exp, q_emb, pos_emb, all_q_emb, all_pos_emb, neg_emb)
            else:
                loss_spec = self.contrastive_loss_fn(logit_scale_exp, emb, all_pos_emb, neg_emb)
                if query_embeddings is not None:
                    loss_spec_q2i = self.contrastive_loss_fn(logit_scale_exp, q_emb, all_pos_emb, neg_emb)

            if labels is not None:
                loss_spec = loss_spec * labels.to(dtype=loss_spec.dtype)
                if query_embeddings is not None:
                    loss_spec_q2i = loss_spec_q2i * labels.to(dtype=loss_q2i.dtype)

            loss += loss_spec.mean() / len(self.mrl_nesting_list)
            if query_embeddings is not None:
                loss_q2i += loss_spec_q2i.mean() / len(self.mrl_nesting_list)

        loss += loss_q2i

        with torch.no_grad():
            temperature = 1.0 / logit_scale_exp.item()
            cosent_temperature = 1.0 / cosent_logit_scale_exp.item()

            if not self.enable_inf_cl or self.use_self_contrast:
                query_emb_all = all_emb
                passage_emb_all = all_pos_emb
            else:
                query_emb_all = concat_all_gather(emb)
                passage_emb_all = concat_all_gather(pos_emb)

            predicted_indices = torch.argmax(query_emb_all @ passage_emb_all.T, axis=1)
            targets = torch.arange(query_emb_all.size(0), device=query_emb_all.device, dtype=torch.long)
            accuracy = torch.mean((predicted_indices == targets).float())
            self.metric_hook['accuracy'].append(accuracy.item())
            self.metric_hook['temperature'].append(temperature)
            if self.cosent_weight:
                self.metric_hook['loss_cosent'].append(loss_cosent.item())
                self.metric_hook['cosent_temperature'].append(cosent_temperature)
            if query_embeddings is not None:
                self.metric_hook['loss_q2i'].append(loss_q2i.item())

        ret = {'loss': loss, 'loss_cosent': loss_cosent.item(), 'loss_q2i': loss_q2i.item()}
        return ret

    def forward(self, **kwargs):
        # q_reps, p_reps = self.get_query_passage_reps(**kwargs)
        # return DROutput(
        #     q_reps=q_reps,
        #     p_reps=p_reps,
        #     logit_scale_exp=self.logit_scale_bias.exp(),
        #     # logit_scale_exp=1.0 / self.temperature_bias,
        # )
        return self.compute_contrastive_loss(**kwargs)


class Qwen3DHNMEmbedder(Qwen3Embedder):
    def __init__(
        self,
        model_name_or_path: str,
        pooler_type: str = 'last',
        do_norm: bool = False,
        truncate_dim: int = 0,
        padding_left: bool = False,
        attn_type: str = 'causal',
        lora_rank: int = 0,
        adapter_path: str = '',
        quantization: bool = True,
        resume_lora: bool = False,
        # use_gradient_checkpointing: bool = True,
        temperature: float = 0.07,
        use_scaling_layer: bool = False,
        mrl_nesting_list: list = [128, 512, 768, 1024],
        enable_inf_cl: bool = False,
        # dynamic hard negative mining params:
        enable_hard_negative_mining: bool = True,
        num_cached_examples: int = 65536,
        num_negs_per_query: int = 256,
        max_length: int = 512,
        relative_margin: float = 0.05,
        momentum: float = 1.0,
        cosent_weight: float = 0.0,
        mask_false_negatives: bool = False,
        use_self_contrast: bool = False,
        fast_decay_temperature: bool = False,
        freeze_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            pooler_type=pooler_type,
            do_norm=do_norm,
            truncate_dim=truncate_dim,
            padding_left=padding_left,
            attn_type=attn_type,
            lora_rank=lora_rank,
            adapter_path=adapter_path,
            quantization=quantization,
            resume_lora=resume_lora,
            # use_gradient_checkpointing=use_gradient_checkpointing,
            temperature=temperature,
            use_scaling_layer=use_scaling_layer,
            mrl_nesting_list=mrl_nesting_list,
            enable_inf_cl=enable_inf_cl,
            cosent_weight=cosent_weight,
            mask_false_negatives=mask_false_negatives,
            use_self_contrast=use_self_contrast,
            fast_decay_temperature=fast_decay_temperature,
            freeze_word_embeddings=freeze_word_embeddings,
            **kwargs,
        )

        self.enable_hard_negative_mining = enable_hard_negative_mining
        self.num_negs_per_query = num_negs_per_query
        self.relative_margin = relative_margin
        self.K = num_cached_examples
        self.momentum = momentum
        self.max_length = max_length

        self.passed_first_batch = False
        self.cache_ready = False
        self.cache_dim = max(self.mrl_nesting_list)

        if enable_hard_negative_mining:
            self.ema_base_model = deepcopy(self.base_model)
            self.ema_base_model.requires_grad_(False)
            self.ema_base_model.eval()
            if self.use_scaling_layer:
                self.ema_scaling_layer = deepcopy(self.scaling_layer)
                self.ema_scaling_layer.requires_grad_(False)
                self.ema_scaling_layer.eval()

        self.register_buffer('queue', torch.randn(self.cache_dim, num_cached_examples))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer(
            'text_queue',
            torch.zeros((num_cached_examples, self.max_length), dtype=torch.long),
        )
        self.register_buffer('text_queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def get_ema_model_embedding(self, input_ids: torch.Tensor, attention_mask: Union[torch.Tensor, None] = None, **kwargs) -> torch.Tensor:
        if input_ids is None:
            return None
        if attention_mask is None:
            attention_mask = create_attention_mask_from_input_ids(input_ids, self.pad_token_id, self.padding_left)
        output = self.ema_base_model(input_ids, attention_mask, return_dict=True, **kwargs)
        emb = self.pooling(output.last_hidden_state, attention_mask)
        if self.use_scaling_layer:
            emb = self.ema_scaling_layer(emb)
        return emb

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys) -> None:
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_text_ids(self, text_ids: torch.Tensor) -> None:
        text_ids = F.pad(text_ids, (0, self.max_length - text_ids.shape[1], 0, 0))
        text_ids = concat_all_gather(text_ids)

        batch_size = text_ids.shape[0]

        ptr = int(self.text_queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue), pad text_ids to max_length
        self.text_queue[ptr : ptr + batch_size] = text_ids
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.text_queue_ptr[0] = ptr

    @torch.no_grad()
    def mine_hard_negatives(self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor):
        if not self.enable_hard_negative_mining:
            return None

        # update the cached model
        self._momentum_update_key_encoder(self.base_model, self.ema_base_model, self.momentum)
        if self.use_scaling_layer:
            self._momentum_update_key_encoder(self.scaling_layer, self.ema_scaling_layer, self.momentum)

        ptr = int(self.queue_ptr)
        # print(f'ptr: {ptr}, cache_ready: {self.cache_ready}, passed_first_batch: {self.passed_first_batch}')

        # check if negatives has been filled for the first time
        if not self.cache_ready:
            if self.passed_first_batch and ptr == 0:
            # if self.passed_first_batch:
                self.cache_ready = True
            else:
                self.passed_first_batch = True

        # sample negatives from the queue
        query_embs = self.get_ema_model_embedding(text_ids)
        query_embs = F.normalize(query_embs[..., : self.cache_dim], dim=-1)
        doc_embs = self.get_ema_model_embedding(text_pos_ids)
        doc_embs = F.normalize(doc_embs[..., : self.cache_dim], dim=-1)
        sim_q2p = torch.einsum('nc,nc->n', [query_embs, doc_embs.to(dtype=query_embs.dtype)])
        sim_q2d = torch.einsum('nc,ck->nk', [query_embs, self.queue.to(dtype=query_embs.dtype)])

        # filter false negatives
        removed_indices = sim_q2d > sim_q2p.repeat(sim_q2d.size(1), 1).T * (1.0 - self.relative_margin)
        sim_q2d.masked_fill_(removed_indices, -1e4)
        _, topk_indices = torch.topk(sim_q2d, self.num_negs_per_query, dim=1, largest=True)
        text_neg_ids = self.text_queue[topk_indices].clone().detach()

        # update the queue
        self._dequeue_and_enqueue(doc_embs)
        self._dequeue_and_enqueue_text_ids(text_pos_ids)

        return text_neg_ids if self.cache_ready else None

    @torch.no_grad()
    def refresh_queue(self, chunck_size=128):
        self.text_queue_ptr[0] = 0
        self.queue_ptr[0] = 0

        world_size, rank = dist.get_world_size(), dist.get_rank()
        global_chunk_size = chunck_size * world_size
        num_iters = self.K // global_chunk_size
        for i in range(num_iters):
            text_ids = self.text_queue[i * global_chunk_size + rank * chunck_size : i * global_chunk_size + (rank + 1) * chunck_size]
            doc_embs = self.get_ema_model_embedding(text_ids)
            doc_embs = F.normalize(doc_embs[..., : self.cache_dim], dim=-1)
            self._dequeue_and_enqueue(doc_embs)

        assert int(self.queue_ptr) == 0

    @torch.no_grad()
    def _momentum_update_key_encoder(self, encoder_q, encoder_k, m=0.0) -> None:
        """
        Momentum update of the key encoder
        """
        if m == 1.0:
            return
        for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

    def forward(self, text_ids, text_pos_ids, **kwargs):
        text_neg_ids = self.mine_hard_negatives(text_ids, text_pos_ids)

        if text_neg_ids is not None:
            bs, num_negs, _ = text_neg_ids.shape
            text_neg_ids = text_neg_ids.view(bs * num_negs, -1)

        loss = self.compute_contrastive_loss(text_ids, text_pos_ids, text_neg_ids, **kwargs)
        logger.info(f'hard negatives used: {0 if text_neg_ids is None else text_neg_ids.shape[0]}, weighted DHNM triplet loss: {loss}')

        return loss
