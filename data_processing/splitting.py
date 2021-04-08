import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse
import sys

import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input path to unsplit data, output dir')
    parser.add_argument('i', type=str, help='path to data')
    parser.add_argument('o', type=str, help='dir to write output - include /')
    parser.add_argument('-c', type=int, default=5, help='count of times to split')
    parser.add_argument('-v', type=float, default=0.15, help='valid size, relative to entire dataset')
    parser.add_argument('-t', type=float, default=0.15, help='test size, relative to entire dataset')
    parser.add_argument('-tvt', type=bool, default=False, help='"train valid test" or just "train valid"')

    if len(sys.argv)<3:
            parser.print_help(sys.stdout)
            sys.exit(1) # missing arguments
    
    args = parser.parse_args()

    data = pd.read_json(args.i, lines=True, orient='records')
    
    if not os.path.exists(args.o):
        os.makedirs(args.o)
    for i in range(args.c):
        if args.tvt:
            train, val = train_test_split(data, test_size=args.v, random_state=i)
            train, test = train_test_split(train, test_size=args.t / (1 - args.v), random_state=i)
            train.to_json(args.o + 'train_{}.jsonl'.format(i), lines=True, orient='records')
            val.to_json(args.o + 'valid_{}.jsonl'.format(i), lines=True, orient='records')
            test.to_json(args.o + 'test_{}.jsonl'.format(i), lines=True, orient='records')
        else:
            train, val = train_test_split(data, test_size=args.t, random_state=i)
            train.to_json(args.o + 'train_{}.jsonl'.format(i), lines=True, orient='records')
            val.to_json(args.o + 'valid_{}.jsonl'.format(i), lines=True, orient='records')

    print('Split complete.')