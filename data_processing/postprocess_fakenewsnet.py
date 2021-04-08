import pandas as pd
from sklearn.metrics import classification_report, f1_score
import json
import numpy as np

NUM_SPLITS = 5
LM_NAME = "elmo"

f1_array = []

all = pd.read_csv('raw_data/gossipcop_all.tsv', sep='\t', lineterminator='\n')

all = all.rename(columns={'tweet':'text'})
three_plus_events = all.event.value_counts().index[all.event.value_counts() >= 3]

for split_num in range(NUM_SPLITS):
    print(split_num)
    df_true = pd.read_json(f'data/gossipcop/test_{split_num}.jsonl', orient='records',lines=True)
    df_preds = pd.read_json(f'preds/gossipcop_{LM_NAME}_{split_num}', orient='records',lines=True)
    df_preds['event'] = df_true.event
    df_preds['true_label'] = df_true.label
    df_preds['label'] = df_preds[0].apply(lambda x: x['label'])
    df_preds = df_preds[df_preds.event.isin(three_plus_events)]
    #print(df_preds.head())
    majority_vote = df_preds.groupby(['event']).agg(lambda x:x.value_counts().index[0])
    f1_array.append(f1_score(majority_vote.true_label.astype(float), majority_vote.label.astype(float), average='macro'))
    #print(majority_vote.head())
    #print(classification_report(majority_vote.true_label.astype(float), majority_vote.label.astype(float)))
print(f1_array)
print(100*round(np.mean(f1_array),3), 100*round(np.std(f1_array),3))


'''
all = pd.read_csv('raw_data/politifact_all.tsv', sep='\t', lineterminator='\n')

all = all[(all.event != 'politifact14920') & (all.event != 'politifact14940')] # remove broken events labeled both ways
all = all.rename(columns={'tweet':'text'})
three_plus_events = all.event.value_counts().index[all.event.value_counts() >= 3]

for split_num in range(NUM_SPLITS):
    print(split_num)
    df_true = pd.read_json(f'data/politifact/test_{split_num}.jsonl', orient='records',lines=True)
    df_preds = pd.read_json(f'preds/politifact_{LM_NAME}_{split_num}', orient='records',lines=True)
    df_preds['event'] = df_true.event
    df_preds['true_label'] = df_true.label
    df_preds['label'] = df_preds[0].apply(lambda x: x['label'])
    df_preds = df_preds[df_preds.event.isin(three_plus_events)]
    #print(df_preds.head())
    majority_vote = df_preds.groupby(['event']).agg(lambda x:x.value_counts().index[0])
    f1_array.append(f1_score(majority_vote.true_label.astype(float), majority_vote.label.astype(float), average='macro'))
    #print(majority_vote.head())
    #print(classification_report(majority_vote.true_label.astype(float), majority_vote.label.astype(float)))
print(f1_array)
print(100*round(np.mean(f1_array),3), 100*round(np.std(f1_array),3))
'''