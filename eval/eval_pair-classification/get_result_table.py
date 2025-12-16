import numpy as np
import pandas as pd
import argparse
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import label_ranking_average_precision_score



parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str)
parser.add_argument("--data_root", type=str, default='test_dataset/')
parser.add_argument("--pos_label", type=int, default=3)
parser.add_argument('--visual', action='store_true')
parser.add_argument('--badcase', action='store_true')

args = parser.parse_args()
print(args)

pos_label = args.pos_label

def compute_metrics(y_true, probs):
    ap = average_precision_score(y_true, probs)

    return dict(
        ap=ap,
    )

matryoshka_dim_list = [128, 256, 512, 1024, 2560]

testset_list = [
'testset'
]


for pos_label in [1,2,3]:
    output_f = open('pair-classification_AP_results_{}'.format(pos_label), 'a+')

    tstep = 'all'
    tdim = 128
    testset_path2res = {}

    def get_textnet_level(textnet_output, level):
        textnet_output_list = textnet_output.split('-')
        textnet_output = '-'.join(textnet_output_list[:level])
        return textnet_output

    filter_set = [
    '媒体账号-个人账号',
    '生活休闲-购物',
    '直播-直播间'
    ]

    filter_set = set(filter_set)

    for i in range(len(testset_list)):
        testset_path = testset_list[i]
        use_new_label = False

        queries, gids, labels, sims = [], [], [], []
        textnets = []
        result_path = os.path.join('./test_result/', '{}_{}.csv'.format(testset_path, args.model_name))

        textnet_filter_set = set()
        query2textnet = {}

        df = pd.read_csv(result_path)
        df.drop_duplicates(subset=['query', 'gid'], inplace=True)
        textnet_filter_count = 0
        nickname_filter_count = 0
        for _, row in df.iterrows():
            query = row['query']
            gid = row['gid']
            label = row['label']
            tk = 'q_d_sim_{}'.format(tdim)
            if tk not in row: continue
            
            qxgid = '{}_{}'.format(query, gid)
            if len(query2textnet) > 0 and query not in query2textnet: 
                textnet_filter_count += 1
                continue
            if query in query2textnet and query2textnet[query] in filter_set: 
                textnet_filter_count += 1
                continue

            cos_sim = row[tk]
            textnet = query2textnet.get(query, '')

            
            queries.append(query)
            gids.append(gid)
            labels.append(label)
            sims.append(cos_sim)
            textnets.append(textnet)

        if len(labels) == 0: continue
        assert len(labels) == len(sims)

        labels = np.array(labels)
        sims = np.array(sims)
        testset_path2res[testset_path] = (labels, sims, queries, gids, textnets)

    output_list = []

    res_list = []
    for testset_path in testset_list:
        labels, sims = testset_path2res[testset_path][:2]
        score2count = {}
        for i in [3, 2, 1, 0]:
            score2count[i] = (labels == i).sum()
        print(testset_path, score2count, score2count[3], len(labels) - score2count[3])

        res = compute_metrics(labels>=pos_label, sims)
        key_list = ['ap']

        output_list += ['{:.4f}'.format(res[tk]) for tk in key_list]
        res_list += [res[tk] for tk in key_list]


    res_list = np.array(res_list).reshape(-1, 1)
    print('res_list', res_list.shape)
    res_list = res_list[:2].mean(0).tolist()

    output_list = ['{:.4f}'.format(tt) for tt in res_list] + output_list

    # print(args.model_name, ','.join(output_list), file = output_f)
    print(args.model_name, file = output_f)
    print(','.join(output_list), file = output_f)