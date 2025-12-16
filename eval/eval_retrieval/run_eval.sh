CKPT_PATH=/mnt/bn/albert-nas-hl/query_generation_sft/Qwen3-Embedding-4B


ADAPTER_PATHES=("/mnt/bn/albert-nas-hl/query_generation_sft/search_qwen_model/iteration_query_model_e1_old_score_e4_new_items_100w_filtered_no_12_data/batch_32_epoch_1_4B_hard_negatives/checkpoint-2000")


RESULT_PATH="./test_result/retrieval_ndcg_result.txt"

len=${#ADAPTER_PATHES[@]}
echo "test for "${len}" models, "${#ADAPTER_PATHES[@]}" ckpts"

for (( i=0; i<$len; i++ )); do
    adapter_path=${ADAPTER_PATHES[$i]}
    echo $adapter_path
    python3 test_ndcg_llm_args_version.py \
    --ckpt_path $CKPT_PATH \
    --adapter_path $adapter_path \
    --test_file "../../../data/retrieval_testset/data.json" \
    --test_label_file "../../../data/retrieval_testset/label.json" \
    --device_id 1 \
    >> $RESULT_PATH
done