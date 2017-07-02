import numpy as np 
import pandas as pd
import os
import pickle
import csv
from collections import Counter

from model.config import Config, Path

try:
    w2v_d = pickle.load(open(os.path.join(Path.interim_input, 'w2v_d.pkl'), 'rb'))
    ch_cnt = pickle.load(open(os.path.join(Path.interim_input, 'ch_cnt.pkl'), 'rb'))
    get_n_common_chs = lambda i: {ch_freq[0] for ch_freq in ch_cnt.most_common(i)}
except:
    raise Exception("Please run python3 -m model.w2v first to generate word2vec model")

def padding(sentence, n):
    return sentence[:n] + '\0' * max(n - len(sentence), 0)

def embed(sentence):
    return [w2v_d[ch] for ch in map(lambda ch: ch if ch in w2v_d else 'UNK', sentence)]

def debed(features, n_common_ch=1000):
    common_chs = get_n_common_chs(n_common_ch)
    return ''.join([max([(word, np.dot(feature, vec)) for word, vec in w2v_d.items() if word in common_chs], key=lambda x: x[1])[0] for feature in features])

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(Path.raw_input, 'raw.csv'))

    df.trans = df.trans.apply(lambda s: padding(s, Config.trans_seq_len)).apply(embed)
    df.origin = df.origin.apply(lambda s: padding(s, Config.origin_seq_len)).apply(embed)

    df.to_pickle(os.path.join(Path.processed_input, "data.pkl"))