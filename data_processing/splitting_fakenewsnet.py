import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.model_selection import train_test_split
import os

NUM_SPLITS = 5

all = pd.read_csv('raw_data/politifact_all.tsv', sep='\t', lineterminator='\n')

all = all[(all.event != 'politifact14920') & (all.event != 'politifact14940')] # remove broken events labeled both ways
all = all.rename(columns={'tweet':'text'})

indices = pd.Series(all.event.unique())

if not os.path.exists('data/politifact'):
    os.makedirs('data/politifact')
for i in range(NUM_SPLITS):
    train_indices, test_indices = train_test_split(indices, test_size =.25)
    val_indices, test_indices = train_test_split(test_indices, test_size = 0.6)


    train = all[all['event'].isin(train_indices)]
    val = all[all['event'].isin(val_indices)]
    test = all[all['event'].isin(test_indices)]

    train = train.sample(frac=1)
    val = val.sample(frac=1)
    test = test.sample(frac=1)

    train.to_json(f'data/politifact/train_{i}.jsonl', orient='records', lines=True)
    val.to_json(f'data/politifact/val_{i}.jsonl', orient='records', lines=True)
    test.to_json(f'data/politifact/test_{i}.jsonl', orient='records', lines=True)

all = pd.read_csv('raw_data/gossipcop_all.tsv', sep='\t', lineterminator='\n')

all = all.rename(columns={'tweet':'text'})

indices = pd.Series(all.event.unique())

if not os.path.exists('data/gossipcop'):
    os.makedirs('data/gossipcop')
for i in range(NUM_SPLITS):
    train_indices, test_indices = train_test_split(indices, test_size =.25)
    val_indices, test_indices = train_test_split(test_indices, test_size = 0.6)


    train = all[all['event'].isin(train_indices)]
    val = all[all['event'].isin(val_indices)]
    test = all[all['event'].isin(test_indices)]

    train = train.sample(frac=1)
    val = val.sample(frac=1)
    test = test.sample(frac=1)

    train.to_json(f'data/gossipcop/train_{i}.jsonl', orient='records', lines=True)
    val.to_json(f'data/gossipcop/val_{i}.jsonl', orient='records', lines=True)
    test.to_json(f'data/gossipcop/test_{i}.jsonl', orient='records', lines=True)