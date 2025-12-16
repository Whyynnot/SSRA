import os
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from transformers.trainer import Trainer


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class NormalTrainer(Trainer):
    def __init__(
        self,
        efficient_save,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.efficient_save = efficient_save

    def save_ckpt_for_sentence_transformers(self, tmp_dir, output_dir, pooling_mode: str = 'mean'):
        '''convert to sentence transformer format'''
        import shutil
        from sentence_transformers import models, SentenceTransformer

        word_embedding_model = models.Transformer(tmp_dir)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
        if os.path.exists(os.path.join(tmp_dir, 'scaling_layer.bin')):
            state_dict = torch.load(os.path.join(tmp_dir, 'scaling_layer.bin'))
            in_features, out_features = state_dict['linear.weight'].shape[1], state_dict['linear.weight'].shape[0]
            scaling_layer = models.Dense(in_features, out_features, bias=True, activation_function=torch.nn.modules.linear.Identity())
            scaling_layer.load_state_dict(state_dict, strict=True)
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model, scaling_layer], device='cpu')
        else:
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
        model.save(output_dir, safe_serialization=False, create_model_card=False)  # do not generate model card, otherwise it will hang
        shutil.rmtree(tmp_dir)

    def _save(self, output_dir: Optional[str] = None, **kwargs):
        '''save the unwrap model'''

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", output_dir)
        unwrap_model = self.model.model if hasattr(self.model, 'model') else self.model.base_model
        if self.is_world_process_zero():
            # first saves to the tmp dir, then converts to sentence-transformer
            tmp_dir = output_dir + '-tmp'
            unwrap_model.save_pretrained(tmp_dir, safe_serialization=self.args.save_safetensors)
            self.tokenizer.save_pretrained(tmp_dir)
            if hasattr(self.model, 'scaling_layer'):
                scaling_layer = {
                    'linear.weight': self.model.scaling_layer.state_dict()['linear.weight'].data.cpu(),
                    'linear.bias': self.model.scaling_layer.state_dict()['linear.bias'].data.cpu(),
                }
                torch.save(scaling_layer, os.path.join(tmp_dir, 'scaling_layer.bin'))
            self.save_ckpt_for_sentence_transformers(tmp_dir, output_dir, 'lasttoken')

    def _save_checkpoint(self, model, trial, metrics=None):
        if self.efficient_save:
            '''only save the model ckpt weights to save disk mem'''
            from transformers.trainer import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)
        else:
            super()._save_checkpoint(model, trial)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        if len(self.model.metric_hook['accuracy']) != 0:
            logs['accuracy'] = sum(self.model.metric_hook['accuracy']) / len(self.model.metric_hook['accuracy'])
            self.model.metric_hook['accuracy'] = []

        if len(self.model.metric_hook['temperature']) != 0:
            logs['temperature'] = sum(self.model.metric_hook['temperature']) / len(self.model.metric_hook['temperature'])
            self.model.metric_hook['temperature'] = []

        if ('loss_cosent' in self.model.metric_hook) and len(self.model.metric_hook['loss_cosent']) != 0:
            logs['loss_cosent'] = sum(self.model.metric_hook['loss_cosent']) / len(self.model.metric_hook['loss_cosent'])
            self.model.metric_hook['loss_cosent'] = []

        if ('cosent_temperature' in self.model.metric_hook) and len(self.model.metric_hook['cosent_temperature']) != 0:
            logs['cosent_temperature'] = sum(self.model.metric_hook['cosent_temperature']) / len(self.model.metric_hook['cosent_temperature'])
            self.model.metric_hook['cosent_temperature'] = []

        if ('loss_q2i' in self.model.metric_hook) and len(self.model.metric_hook['loss_q2i']) != 0:
            logs['loss_q2i'] = sum(self.model.metric_hook['loss_q2i']) / len(self.model.metric_hook['loss_q2i'])
            self.model.metric_hook['loss_q2i'] = []

        super().log(logs, start_time)
