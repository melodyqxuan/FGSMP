import argparse
import numpy as np
import pandas as pd
import math, os, time, json, datetime, copy, string
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

import torch

import nltk
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords

parser = argparse.ArgumentParser(description='Data building logistics')
parser.add_argument('--build-train-valid', action='store_true', help='build training and testing dataset from scratch')
parser.add_argument('--build-pos', action='store_true', help='build POS tagging mappings, stored in JSON format')
args = parser.parse_args()

LAST_DAY_FLAG = '09:10:00'
OUT_OF_TRADE_FLAG = '14:50:00'
ONE_DAY = datetime.timedelta(days=1)
UNKNOWN = '<UNK>'
TICKERS = os.listdir('train')

class BuildPOS:
    def __init__(self):
        self.n_pos = 1
        self.pos2index = {'<PAD>': 0}
        self.index2pos = {0: '<PAD>'}

    def addPOS(self, POS_arr):
        for pos in POS_arr:
            if pos not in self.pos2index:
                self.pos2index[pos] = self.n_pos
                self.index2pos[self.n_pos] = pos
                self.n_pos += 1

def build_pos(train_root, valid_root):
    files = [os.path.join(train_root, ticker, i) for ticker in TICKERS for i in os.listdir(os.path.join(train_root, ticker))] + \
            [os.path.join(valid_root, ticker, i) for ticker in TICKERS for i in os.listdir(os.path.join(valid_root, ticker))]
    POS = BuildPOS()
    for path in files:
        with open(os.path.join(path, 'seq.json'), 'r') as f:
            POS.addPOS(json.load(f)['pos'])
    with open('pos2index.json', 'w') as f:
        json.dump(POS.pos2index, f)
    with open('index2pos.json', 'w') as f:
        json.dump(POS.index2pos, f)
    return POS

def preprocess(text):
    text = text.lower()
    text_p = "".join([char for char in text if char not in string.punctuation])
    words = word_tokenize(text_p)
    stop_words = stopwords.words('english')
    filtered_words = [word for word in words if word not in stop_words]
    pos = pos_tag(filtered_words)
    return filtered_words, pos

def build_vocab(path_to_event = 'news.json'):
    with open(path_to_event, 'r') as f:
        data = json.load(f)
    vocab = Counter()
    for ticker, news_arr in data.items():
        for news in news_arr:
            filtered_title, _ = preprocess(news['title'])
            filtered_body, _  = preprocess(news['text'])
            for corpus in [filtered_title, filtered_body]:
                for word in corpus:
                    vocab[word] += 1
    out = set()
    for word, freq in vocab.items():
        if freq >= 5:
            out.add(word)
    return out

def train_valid_split(path_to_event = 'news.json', path_to_stock = 'historical_price'):

    def build_ticker():
        tickers = os.listdir(path_to_stock)
        for i, ticker_str in enumerate(tickers):
            tickers[i] = ticker_str.split('_')[0]
        return set(tickers)

    def get_fname(ticker):
        ticker_files = os.listdir(path_to_stock)
        for fname in ticker_files:
            if ticker in fname:
                return fname
        raise ValueError('Invalid ticker:', ticker)

    def build_seq_and_pos(news):
        seq, pos = [], []
        for corpus in [news['title'], news['text']]:
            corpus_processed, corpus_pos = preprocess(corpus)
            assert len(corpus_processed) == len(corpus_pos)
            for i in range(len(corpus_processed)):
                w, p = corpus_processed[i], corpus_pos[i][1]
                word = w if w in vocab else UNKNOWN
                seq.append(word)
                pos.append(p)
        return seq, pos

    def build_dataset(ticker, stock_df, news_arr, path):
        '''In our crude preprocessing stage, we create movement label by computing fetching
           next business day's open price. We simply ignore scenarios in which non-trading holidays
           and business days overlap
        '''
        if not os.path.exists(path): os.mkdir(path)
        if not os.path.exists(os.path.join(path, ticker)): os.mkdir(os.path.join(path, ticker))

        for ind, news in enumerate(news_arr):
            date, time = news['pub_time'].split(' ')
            time = time.split('+')[0]
            if time <= LAST_DAY_FLAG or time >= OUT_OF_TRADE_FLAG:
                yesterday = datetime.datetime.fromisoformat(date) - ONE_DAY
                while yesterday.weekday() >= 5:
                    yesterday -= ONE_DAY
                yesterday = yesterday.strftime('%Y-%m-%d')
                stock = stock_df.loc[stock_df['date'] == yesterday]
            else:
                stock = stock_df.loc[(stock_df['date'] == date) & (stock_df['time'] <= time)]

            tomorrow = datetime.datetime.fromisoformat(date) + ONE_DAY
            while tomorrow.weekday() >= 5:
                tomorrow += ONE_DAY
            tomorrow = tomorrow.strftime('%Y-%m-%d')
            stock_tomorrow = stock_df.loc[stock_df['date'] == tomorrow]


            if len(stock) <= 13 or len(stock_tomorrow) == 0:
                continue

            y = int(stock_tomorrow.iloc[0]['o'] > stock.iloc[-1]['c'])
            stock_processed = []
            stock = stock.drop(['date', 'time'], axis = 1).to_numpy()
            scaler = MinMaxScaler()
            stock_cp = copy.deepcopy(stock)
            scaler.fit(stock_cp)
            stock_cp = scaler.transform(stock_cp)
            for i in range(11, len(stock), 10):
                history = stock[i-11:i-1].reshape(60)
                chunk = stock[i-10:i].reshape(60)
                chunk_scaled = stock_cp[i-10:i].reshape(60)
                change = (chunk - history) / (history + 1e-6)
                stock_processed.append(np.concatenate((chunk_scaled, change), axis=0))
            stock_processed = np.array(stock_processed)

            seq, pos = build_seq_and_pos(news)
            save = os.path.join(path, ticker, str(ind))
            if not os.path.exists(save): os.mkdir(save)
            np.save(os.path.join(save, 'stock.npy'), stock_processed)
            with open(os.path.join(save, 'seq.json'), 'w') as f:
                json.dump({'text': seq, 'pos': pos, 'tgt': y}, f)

    print('Building vocabulary...')
    start = time.time()
    vocab = build_vocab(path_to_event)
    print('Vocabulary building completed in {:.4f}sec'.format(time.time()-start))

    tickers = build_ticker()

    with open(path_to_event, 'r') as f:
        data = json.load(f)

    with tqdm(total=len(data)) as pbar:
        for ticker, news_list in data.items():
            if ticker not in tickers: continue
            news_sorted = sorted(news_list, key = lambda news: news['pub_time'])
            split_ind = int(len(news_sorted) * 0.7)
            news_train, news_valid = news_sorted[:split_ind], news_sorted[split_ind:]

            stock = pd.read_csv(os.path.join(path_to_stock, get_fname(ticker)), index_col = 0, header = 0)
            stock[['date', 'time']] = stock['t'].str.split(' ', expand=True)
            stock = stock.drop(['t', 'n'], axis = 1)
            build_dataset(ticker, stock, news_train, './train')
            build_dataset(ticker, stock, news_valid, './valid')
            pbar.update()

if args.build_train_valid:
    train_valid_split()

if args.build_pos:
    build_pos('train', 'valid')
