#!/bin/bash

ROOT=$PWD
export PYTHONPATH=$ROOT:${PYTHONPATH}


# Arnold Parameter
GPUS_PER_NODE=${ARNOLD_WORKER_GPU}
WORLD_SIZE=${ARNOLD_WORKER_NUM}
RANK=${ARNOLD_ID}
MASTER_ADDR=${METIS_WORKER_0_HOST}
PORTS=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
MASTER_PORT=${PORTS[0]}
if [ -z "$WORLD_SIZE" ]; then
    GPUS_PER_NODE=1
    WORLD_SIZE=1
    RANK=0
    MASTER_ADDR=127.0.0.1
    MASTER_PORT=6000
fi

# Hyper Parameter Start
# MODEL_NAME_OR_PATH=/mnt/bn/yjy-search-dev-hl/pretrained/Qwen3-Embedding-0.6B
MODEL_NAME_OR_PATH=/mnt/bn/albert-nas-hl/query_generation_sft/Qwen3-Embedding-4B

# ADAPTER_PATH=/mnt/bn/yjy-search-dev-hl/query2item_output/stage1_qwen3_pretrain_infcl_lora/checkpoint-90000
EPOCHS=1
BATCH_SIZE=32
TEMPERATURE=3e-2
LR=1e-5
LR_SCHEDULER_TYPE=cosine
# LR_SCHEDULER_KWARGS="{\"min_lr_rate\": 1e-2}"
WEIGHT_DECAY=1e-3
WARMUP_RATIO=1e-2
WARMUP_STEPS=300
NEG_NUM=3
DS_PATH=$ROOT/ds_config_zero2.json
MAX_LENGTH=512
MRL_NESTING_LIST=(128 512 1024 1536 2048 2560)


ROOT_DIRS=(
# ../../data/search_relevance_trainset/trainset
/mnt/hdfs/query2item/synthetic_data/search/renshen_sft_data/data/iteration_query_model_e1_old_score_e4_new_items_100w_filtered_no_12_data

)

OUTPUT_PATH_PREFIX=./trained_model

OUTPUT_DIR=(
$OUTPUT_PATH_PREFIX/baseline


)
# Hyper Parameter End


model_args=(
    "--model_name_or_path" $MODEL_NAME_OR_PATH
    # "--adapter_path" $ADAPTER_PATH
    "--max_length=$MAX_LENGTH"
    "--temperature=$TEMPERATURE"
    "--use_scaling_layer=False"
    "--mrl_nesting_list" "${MRL_NESTING_LIST[@]}"
    "--lora_rank=32"
    "--quantization=False"
    "--resume_lora=False"
    "--enable_hard_negative_mining=True"
    "--num_cached_examples=65536"
    # "--num_cached_examples=1024"
    "--num_negs_per_query=$NEG_NUM"
    "--enable_inf_cl=False"
    "--cosent_weight=0.0"
    "--mask_false_negatives=True"
    "--use_self_contrast=True"
    "--fast_decay_temperature=True"
    # "--freeze_word_embeddings=True"
)

data_args=(
    "--root_dirs" "${ROOT_DIRS[@]}"
    "--neg_num=$NEG_NUM"
)

train_args=(
    "--bf16"
    "--gradient_checkpointing=True"
    "--output_dir=$OUTPUT_DIR"
    "--num_train_epochs=$EPOCHS"
    "--dataloader_num_workers=8"
    "--batch_size=$BATCH_SIZE"
    "--learning_rate=$LR"
    "--lr_scheduler_type=$LR_SCHEDULER_TYPE"
    # "--lr_scheduler_kwargs='$LR_SCHEDULER_KWARGS'"
    "--weight_decay=$WEIGHT_DECAY"
    # "--warmup_ratio=$WARMUP_RATIO"
    "--warmup_steps=$WARMUP_STEPS"
    "--deepspeed=$DS_PATH"
    "--logging_steps=1"
    "--save_safetensors=False"
    "--report_to=wandb"
    "--save_strategy=steps"
    "--save_steps=500"
    "--save_total_limit=20"
    "--per_device_train_batch_size=1"
    "--efficient_save=False"
    "--ignore_data_skip=True"
)

all_args=("${model_args[@]}" "${data_args[@]}" "${train_args[@]}")


export LAUNCHER="python3 -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

export CMD=" \
    $ROOT/train_qwen3_npu.py \
    "

echo $CMD

bash -c "$LAUNCHER $CMD ${all_args[*]}"
