import json
from typing import List

import pytrec_eval
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import torch
try:
    import torch_npu # 注意：torch 和 torch_npu的版本是强对应的，不要更改torch版本，在安装依赖库时要特别注意
    from torch_npu.contrib import transfer_to_npu # 执行替换操作
except:
    print('no npu dependencies found, maybe using cuda...')

import argparse

args = argparse.ArgumentParser()
args.add_argument('--ckpt_path', type=str, default='/mnt/bn/albert-nas-hl/query_generation_sft/Qwen3-Embedding-4B')
args.add_argument('--adapter_path', nargs='+', type=str, default='')
args.add_argument('--test_file', type=str, default='/mnt/bn/albert-nas-hl/query_generation_sft/evaluation/search/retrieval_testset_pair_filtered.json')
args.add_argument('--test_label_file', type=str, default='/mnt/bn/albert-nas-hl/query_generation_sft/evaluation/search/20250430_renshen_retrieval_gt_all_pair_filtered.json')
args.add_argument('--device_id', type=int, default=2)

args = args.parse_args()
CKPT_PATH = args.ckpt_path
ADAPTER_PATH = args.adapter_path
data_path = args.test_file
label_path = args.test_label_file
device_id = args.device_id

print(f'CKPT_PATH: {CKPT_PATH}')
print(f'ADAPTER_PATH: {ADAPTER_PATH}')
print(f'data_path: {data_path}')
print(f'label_path: {label_path}')
print(f'device_id: {device_id}')


if torch_npu.npu.is_available():
    device = f'npu:{device_id}'
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_SEQ_LENGTH = 512

# PROMPT = 'Instruct: Given a web search query, retrieve relevant Douyin clips in Title;OCR;ASR format that answer the query\nQuery:'
# DOC_TEMPLATE = 'Douyin clip:\nTitle:{};\nOCR:{};\nASR:{}'
PROMPT = 'Instruct: Given a web search query, retrieve relevant Douyin clips in Title;Video_text format that answer the query\nQuery:'
DOC_TEMPLATE = 'Douyin clip:\nTitle:{};\nVideo_text:{}'
LABEL_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
}

print(f'CKPT: {CKPT_PATH}, MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}')
print(f'PROMPT: {PROMPT}, DOC_TEMPLATE: {DOC_TEMPLATE}')

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer_kwargs={"padding_side": "left"}
prompt_name = 'query'

model = SentenceTransformer(
    model_name_or_path=CKPT_PATH,
    local_files_only=True,
    device=device,
    tokenizer_kwargs=tokenizer_kwargs,
)

if ADAPTER_PATH:
    if isinstance(ADAPTER_PATH, str):
        model.load_adapter(ADAPTER_PATH)
    else:
        adapter_names = []
        for i, adapter_path in enumerate(ADAPTER_PATH):
            model.load_adapter(adapter_path, adapter_name=f'adapter_{i}')
            adapter_names.append(f'adapter_{i}')
        model.set_adapter(adapter_names)

model.eval()
model.max_seq_length = MAX_SEQ_LENGTH
model.prompts['query'] = PROMPT


def get_text_embedding(text, prompt_name=None):
    emb = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=True,
        prompt_name=prompt_name,
    )
    return emb


if __name__ == '__main__':
    # limit_q_num = 10
    limit_q_num = None
    qrel = {}
    known_items = set()
    min_list_size, max_list_size = float('inf'), float('-inf')

    doc_id2text = {}
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'item_id' not in data:
                continue
            doc_id = str(data['item_id'])
            doc_title = data['title']
            doc_video_text = data['video_text']
            doc_text = DOC_TEMPLATE.format(doc_title, doc_video_text)
            doc_id2text[doc_id] = doc_text

    with open(label_path, 'r') as f:
        for i, line in enumerate(f):
            if limit_q_num and i >= limit_q_num:
                break
            data = json.loads(line)
            query = data['query']
            doc_rel = {str(k): LABEL_MAP[int(v)] for k, v in data['gt'] if (str(k) in doc_id2text and int(v) in LABEL_MAP)}
            if len(doc_rel) < 10:
                continue
            min_list_size = min(min_list_size, len(doc_rel))
            max_list_size = max(max_list_size, len(doc_rel))
            known_items.update(doc_rel.keys())
            qrel[query] = doc_rel

    doc_id2text = {k: v for k, v in doc_id2text.items() if k in known_items}
    all_docs = set(doc_id2text.values())

    all_queries = list(qrel)
    all_docs = list(all_docs)
    print(f'min_list_size: {min_list_size}; max_list_size: {max_list_size}')
    print(f'len(all_queries): {len(all_queries)}; len(all_docs): {len(all_docs)}')

    query_embs = get_text_embedding(all_queries, prompt_name=prompt_name)
    doc_embs = get_text_embedding(all_docs, prompt_name=None)
    print(f'query_embs.shape: {query_embs.shape}; doc_embs.shape: {doc_embs.shape}')

    query2emb = {}
    for query, query_emb in zip(all_queries, query_embs):
        query2emb[query] = query_emb

    doc_text2emb = {}
    for doc_text, doc_emb in zip(all_docs, doc_embs):
        doc_text2emb[doc_text] = doc_emb

    doc_id2emb = {}
    for doc_id, doc_text in doc_id2text.items():
        doc_id2emb[doc_id] = doc_text2emb[doc_text]

    run = {}
    # qrel: key->query; value->dict{key->doc_id; value->relevance}
    for query, doc_dict in tqdm(qrel.items(), total=len(qrel)):
        query_emb = query2emb[query]
        doc_ids_q = doc_dict.keys()
        doc_embs_q = np.vstack([doc_id2emb[id] for id in doc_ids_q])
        scores = np.dot(doc_embs_q, query_emb).tolist()
        run[query] = {id: score for id, score in zip(doc_ids_q, scores)}

    metrics = {'ndcg_cut_5', 'ndcg_cut_10'}
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)
    results = evaluator.evaluate(run)

    mean_ndcg = {
        'NDCG@5': sum(query_results['ndcg_cut_5'] for query_results in results.values()) / len(results),
        'NDCG@10': sum(query_results["ndcg_cut_10"] for query_results in results.values()) / len(results),
    }

    print(mean_ndcg)
