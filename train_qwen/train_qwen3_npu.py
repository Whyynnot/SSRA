import re
import os
import yaml
import torch
try:
    import torch_npu      # 注意：torch 和 torch_npu的版本是强对应的，不要更改torch版本，在安装依赖库时要特别注意
    from torch_npu.contrib import transfer_to_npu # 执行替换操作
except:
    print('using cuda...')

from pathlib import Path
from transformers.trainer import logger, Optional
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets, load_from_disk
from dataclasses import asdict
from transformers import AutoTokenizer, HfArgumentParser, TrainerCallback  # type: ignore
from transformers.trainer import Trainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import set_verbosity_info

from piccolo.arguments import ModelArguments, DataArguments, STETrainingArguments, Qwen3ModelArguments, Qwen3DHNModelArguments
from piccolo.data_parquet import OnlyStage1Collator, OnlyStage1WeightedCollator, Query2ItemGroupCollator, DistIterableDataset  # not open source yet
from piccolo.cruise.utils import parse_data_source  # not open source yet
from piccolo.model_qwen3_with_loss import Qwen3DHNMEmbedder
from trainer.normal_trainer import NormalTrainer
import numpy as np


# set_verbosity_info()
if torch.__version__ > '2.5.0':
    allow_list = [np.core.multiarray._reconstruct, np.ndarray, np.dtype]
    allow_list += [type(np.dtype(np.uint32))]
    torch.serialization.add_safe_globals(allow_list)


class MyCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, train_dataloader, **kwargs):
        #train_dataloader.dataset.create_or_refresh_data()
        train_dataloader.dataset.need_shuffle = True


def get_resume_ckpt(output_dir):
    try:
        ret = get_last_checkpoint(output_dir)
        return ret
    except:
        pass
    return None


def get_global_step(ckpt_path):
    if isinstance(ckpt_path, str) and ckpt_path:
        try:
            pattern = r"checkpoint-(\d+)"
            file_name = os.path.basename(ckpt_path)
            match = re.search(pattern, file_name)
            if match:
                step = match.group(1)
                step = int(step)
                return step
        except:
            pass
    return 0


def main():
    parser = HfArgumentParser((Qwen3DHNModelArguments, DataArguments, STETrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: Qwen3DHNModelArguments
    data_args: DataArguments
    training_args: STETrainingArguments

    # set for using parquet dataloader
    training_args.accelerator_config.dispatch_batches = False

    # DataLoader
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side='left', trust_remote_code=True)

    print('data_args', data_args)
    print('train_args', training_args)

    all_parquet = []
    for tpath in data_args.root_dirs:
        if not tpath.endswith('parquet'):
            tpath = os.path.join(tpath, '*.parquet')
        file_list = parse_data_source(tpath)
        all_parquet += file_list[0]

    #print(all_parquet)
    resume_ckpt_path = get_resume_ckpt(training_args.output_dir)
    resume_step = get_global_step(resume_ckpt_path)

    train_dataset = DistIterableDataset(all_parquet, batch_size=data_args.batch_size, url_format='parquet', shuffle=True, resume_step=resume_step)
    print(f'len(train_dataset): {len(train_dataset)}, batch_size: {data_args.batch_size}, resume_step: {resume_step}')

    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    q_instruction = get_detailed_instruct('Given a web search query, retrieve relevant Douyin clips in Title;OCR;ASR format that answer the query', '')
    d_instruction = ''
    text_template = 'Douyin clip:\nTitle:{};\nOCR:{};\nASR:{}'

    q_instruction_len = len(tokenizer(text=q_instruction, padding=False, truncation=False, add_special_tokens=True)['input_ids'])
    d_instruction_len = len(tokenizer(text=d_instruction, padding=False, truncation=False, add_special_tokens=True)['input_ids'])
    print(f'q_instruction_len: {q_instruction_len}; d_instruction_len: {d_instruction_len}')

    data_collator = OnlyStage1WeightedCollator(
        tokenizer=tokenizer,
        max_length=model_args.max_length,
        query_key='text',
        text_keys=['text_pair'],
        text_prompts=['{}'],
        text_template=text_template,
        q_instruction=q_instruction,
        d_instruction=d_instruction,
    )

    # Model
    model = Qwen3DHNMEmbedder(
        model_name_or_path=model_args.model_name_or_path,
        pooler_type=model_args.pooler_type,
        do_norm=model_args.do_norm,
        truncate_dim=model_args.truncate_dim,
        padding_left=model_args.padding_left,
        attn_type=model_args.attn_type,
        # attn_implementation=model_args.attn_implementation,
        lora_rank=model_args.lora_rank,
        adapter_path=model_args.adapter_path,
        quantization=model_args.quantization,
        resume_lora=model_args.resume_lora,
        max_length=model_args.max_length,
        temperature=model_args.temperature,
        use_scaling_layer=model_args.use_scaling_layer,
        mrl_nesting_list=model_args.mrl_nesting_list,
        enable_inf_cl=False,
        # dynamic hard negative mining
        enable_hard_negative_mining=model_args.enable_hard_negative_mining,
        num_cached_examples=model_args.num_cached_examples,
        num_negs_per_query=model_args.num_negs_per_query,
        relative_margin=0.05,
        momentum=1.0,
        cosent_weight=model_args.cosent_weight,
        mask_false_negatives=model_args.mask_false_negatives,
        use_self_contrast=model_args.use_self_contrast,
        fast_decay_temperature=model_args.fast_decay_temperature,
        freeze_word_embeddings=model_args.freeze_word_embeddings,
    )
    model.pad_token_id = tokenizer.pad_token_id

    # Trainer
    trainer = NormalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[MyCallback],
        efficient_save=training_args.efficient_save,
    )

    # save training info
    if trainer.is_world_process_zero():
        Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(training_args.output_dir, 'parameters')).mkdir(parents=True, exist_ok=True)
        
    trainer.train(resume_from_checkpoint=True if resume_ckpt_path else False)

    # save parameter and model at the end
    if trainer.is_world_process_zero():
        trainer.save_model(training_args.output_dir, _internal_call=True)
        ## save parameter
        parameter_dict = {'model_args': asdict(model_args), 'data_args': asdict(data_args), 'train_args': asdict(training_args)}
        Path(os.path.join(training_args.output_dir, 'parameters')).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(training_args.output_dir, 'parameters', 'param.yaml'), 'w') as yaml_file:
            yaml.dump(parameter_dict, yaml_file)


if __name__ == "__main__":
    main()
