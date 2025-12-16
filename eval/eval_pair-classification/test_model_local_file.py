import json
import argparse
import os
import time
import sys    

import torch
import torch.nn.functional as F
import random

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
try:
    import torch_npu # 注意：torch 和 torch_npu的版本是强对应的，不要更改torch版本，在安装依赖库时要特别注意
    from torch_npu.contrib import transfer_to_npu # 执行替换操作
except:
    print('no npu dependencies found, maybe using cuda...')

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--adapter_path", type=str, default='')
parser.add_argument("--test_file", type=str)
parser.add_argument("--batch_size", type=int, default = 128000)
parser.add_argument("--ocr_key", type=str, default = 'attr_ocr')
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--seed", type=int, default=52)
parser.add_argument("--num_samples", type=int, default=5000000)

args = parser.parse_args()
print(f'The args are {args=}')

MAX_SEQ_LENGTH = 512
PROMPT = 'Instruct: Given a web search query, retrieve relevant Douyin clips in Title;Video_text format that answer the query\nQuery:'
TEXT_TEMPLATE = 'Douyin clip:\nTitle:{};\nVideo_text:{}'


if torch_npu.npu.is_available():
    device = f'npu:{args.device_id}'    
else:
    device = f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu'

print(f'device is {device}')

model_name = args.model_name
model_path = args.model_path
suffix = '' if args.ocr_key == 'attr_ocr' else '_' + args.ocr_key

tmp_test_path = args.test_file
tmp_save_path = os.path.basename(tmp_test_path)[:-5] + f'_{model_name}{suffix}.csv'
tmp_save_path = os.path.join('test_result', tmp_save_path)

if os.path.exists(tmp_save_path):
    print(f"{tmp_save_path} 已存在，程序退出。")
    sys.exit(0)

model_kwargs = {}
if 'qwen' in model_path or 'Qwen' in model_path:
    model_kwargs['tokenizer_kwargs'] = {"padding_side": "left"}
    print('model_kwargs', model_kwargs)
model = SentenceTransformer(
    model_name_or_path = model_path,
    local_files_only=True,
    device=device,
    **model_kwargs,
)
if args.adapter_path:
    model.load_adapter(args.adapter_path)
model.eval()
model.max_seq_length = MAX_SEQ_LENGTH

if PROMPT:
    model.prompts['query'] = PROMPT


def get_simi(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

matryoshka_dim_list = [128, 256, 512, 1024, 2560]
if 'qwen' in model_path or 'Qwen' in model_path:
    matryoshka_dim_list = [128, 256, 512, 1024, 2560]



res = []
test_path = args.test_file

time_st = time.time()
with open(test_path, 'r') as f:
    data_list = []
    for idx, line in enumerate(f.readlines()):
        data = json.loads(line.strip())
        query = data.pop('query')
        gid = data.pop('gid')
        label = data.pop('label')

        if not isinstance(query, str) or len(query) == 0:
            continue

        title, video_text = data['title'], data['video_text']
        title = title if isinstance(title, str) else ''

        if len(title) == 0 and len(video_text) == 0:
            print('title and video_text are empty')
            continue

        doc = TEXT_TEMPLATE.format(title, video_text)

        data_list.append( (query, gid, label, doc) )

    # 随机采样data_list中的数据，要保证每个label的取值都对应500条数据
    random.seed(args.seed)
    random.shuffle(data_list)

    label_dict = {}
    for tt in data_list:
        label = tt[2]
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append(tt)


    data_list = []
    for label in label_dict:
        data_list.extend(label_dict[label][:args.num_samples])

    doc_len = [len(tt[3]) for tt in data_list]
    for start_idx in tqdm(range(0, len(data_list), args.batch_size)):
        end_idx = min(start_idx + args.batch_size, len(data_list))

        query_list = [tt[0] for tt in data_list[start_idx:end_idx]]
        gid_list = [tt[1] for tt in data_list[start_idx:end_idx]]
        label_list = [tt[2] for tt in data_list[start_idx:end_idx]]
        doc_list = [tt[3] for tt in data_list[start_idx:end_idx]]
        if 'Qwen' in model_path or 'qwen' in model_path:
            q_emb = model.encode(query_list, 
                                       prompt_name = 'query',
                                       normalize_embeddings=False, show_progress_bar=True, device=device, convert_to_tensor=True, 
                                       )
            d_emb = model.encode(doc_list, normalize_embeddings=False, show_progress_bar=True, device=device, convert_to_tensor=True, 
                                 )
        else:
            q_emb = model.encode(query_list, normalize_embeddings=False, device=device, convert_to_tensor=True)
            d_emb = model.encode(doc_list, normalize_embeddings=False, device=device, convert_to_tensor=True,)

        simi_dict = {}
        emb_size = q_emb.shape[-1]
        for tdim in matryoshka_dim_list:
            if tdim > emb_size: break
            q_d_sim = torch.cosine_similarity(q_emb[:, :tdim], d_emb[:, :tdim])
            simi_dict['q_d_sim_{}'.format(tdim)] = q_d_sim.tolist()

        for i in range(end_idx - start_idx):
            temp_dict = dict(
                query=query_list[i],
                gid=gid_list[i],
                label=label_list[i]
            )
            for tdim in matryoshka_dim_list:
                if tdim > emb_size: break
                temp_dict['q_d_sim_{}'.format(tdim)] = simi_dict['q_d_sim_{}'.format(tdim)][i]

            res.append(temp_dict)


save_path = os.path.basename(test_path)[:-5] + f'_{model_name}{suffix}.csv'
save_path = os.path.join('test_result', save_path)
pd.DataFrame(res).to_csv(save_path, index=False)
end_time = time.time()
print(args.batch_size, end_time - time_st)

