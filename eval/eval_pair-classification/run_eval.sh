if [ ! -d "./test_result" ]; then
    mkdir ./test_result
fi

data_root="../../../data/pair-classification_testset"
echo $data_root

model_path='/mnt/bn/albert-nas-hl/query_generation_sft/Qwen3-Embedding-4B'

model_name=('iteration_query_model_e1_old_score_e4_new_items_100w_filtered_hard_2000')
adapter_path=('/mnt/bn/albert-nas-hl/query_generation_sft/search_qwen_model/iteration_query_model_e1_old_score_e4_new_items_100w_filtered/batch_32_epoch_1_4B_hard_negatives/checkpoint-2000')

DEVICE_ID=1


testset_list=('testset')

len=${#model_name[@]}
len_test=${#testset_list[@]}
echo 'test for '${len}' models, '${#adapter_path[@]}' ckpts'
for (( i=0; i<$len; i++ )); do
	for (( j=0; j<$len_test; j++ )); do
	    tmodel_name=${model_name[$i]}
	    tmodel_path=${model_path}
		tadapter_path=${adapter_path[$i]}
	    ttestset_path=${testset_list[$j]}
	    echo $tmodel_name $tmodel_path $ttestset_path
	    python3 -u test_model_local_file.py --model_name $tmodel_name --model_path $tmodel_path --adapter_path $tadapter_path --test_file ${data_root}/${ttestset_path}.json --device_id ${DEVICE_ID} --seed 52 --num_samples 60
	done
        python get_result_table.py $tmodel_name --data_root ${data_root}
done

