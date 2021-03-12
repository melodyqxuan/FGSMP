import os, time, argparse, json
import tqdm
import numpy as np

import torch
import torch.nn as nn
from model import MSSPM

parser = argparse.ArgumentParser(description='(M)SSPM training logistics')
parser.add_argument('--train-root', default='train', type=str, help='training dataset location')
parser.add_argument('--valid-root', default='valid', type=str, help='validation dataset location')
parser.add_argument('--mode', default=0, type=int, help='training mode. Options from 0, 1, and 2') #0, 1, 2
parser.add_argument('--hidden-dim', default=256, type=int, help='number of hidden dimensions for the Bi-LSTM layers')
parser.add_argument('--sent-encoder-layers', default=3, type=int, help='number of hidden layers for sentence Bi-LSTM encoders')
parser.add_argument('--stock-encoder-layers', default=3, type=int, help='number of hidden layers for stock Bi-LSTM encoders')
parser.add_argument('--num-heads', default=4, type=int, help='number of self-attention heads')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout probability for Bi-LSTMs')
parser.add_argument('--lmbd', default=0.5, type=float, help='weight for the CRF weight')

parser.add_argument('--lr', default=1e-2, type=float, help='optimizer learning rate')
parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
parser.add_argument('--bsize', default=32, type=int, help='batch size')
parser.add_argument('--nepochs', default=50, type=int, help='number of training epochs')

parser.add_argument('--evaluate', action='store_true', help='whether to perform model evaluation')

args = parser.parse_args()

TICKERS = os.listdir('train')

def build_dataset(root):
    with open('pos2index.json', 'r') as f:
        pos2index = json.load(f)
    files = [os.path.join(root, ticker, i) for ticker in TICKERS \
             for i in os.listdir(os.path.join(root, ticker))]
    news, stocks, pos, labels = [], [], [], []
    for path in files:
        with open(os.path.join(path, 'seq.json'), 'r') as f:
            seq_label_dict = json.load(f)
            news.append(seq_label_dict['text'])
            pos.append([pos2index[p] for p in seq_label_dict['pos']])
            assert len(pos[-1]) == len(news[-1])
            labels.append(seq_label_dict['tgt'])
        stocks.append(np.load(os.path.join(path, 'stock.npy')))
    news = np.array(news, dtype = object)
    stocks = np.array(stocks, dtype = object)
    pos = np.array(pos, dtype = object)
    labels = np.array(labels, dtype = np.int)
    return news, stocks, pos, labels

def batch_sampler(idx, news, stocks, pos, labels):

    def pad_sequence(seq_list):
        out = []
        seq_lens = [len(seq) for seq in seq_list]
        max_length = max(seq_lens)
        if isinstance(seq_list[0], list):
            seq_shape = (max_length,)
        else:
            seq_shape = (max_length, seq_list[0].shape[-1])

        for i, seq_i in enumerate(seq_list):
            seq = torch.zeros(seq_shape)
            seq[:len(seq_i)] = torch.Tensor(seq_i)
            out.append(seq)

        out = torch.stack(out).to(torch.float32)
        return out, seq_lens

    stocks_out, stock_lens = pad_sequence(list(stocks[idx]))
    pos_out, pos_lens = pad_sequence(list(pos[idx]))
    pos_out = pos_out.to(torch.long)
    dict_out = {'news': list(news[idx]), 'sent_lens': pos_lens, \
                'stocks': stocks_out, 'stock_lens': stock_lens, \
                'pos': pos_out, 'labels': torch.from_numpy(labels[idx]).to(torch.float32)}
    return dict_out

news_train, stocks_train, pos_train, labels_train = build_dataset(args.train_root)
news_valid, stocks_valid, pos_valid, labels_valid = build_dataset(args.valid_root)
num_train = len(news_train)
num_valid = len(news_valid)

event_size = 38
model = MSSPM(args.hidden_dim, args.sent_encoder_layers, args.stock_encoder_layers, \
              args.num_heads, args.dropout, args.mode, event_size = event_size)
num_params = sum(p.numel() for p in model.parameters())
print('Number of Parameters: {}'.format(num_params))
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

def train_epoch(curr_epoch, model, criterion, optimizer):
    model.train()
    ind = np.random.permutation(num_train)
    minibatch_iter = tqdm.tqdm(range(0, num_train, args.bsize), desc = f"(Epoch {curr_epoch}) Minibatch")
    #for i in range(0, num_train, args.bsize):
    for i in minibatch_iter:
        optimizer.zero_grad()
        input_dict = batch_sampler(ind[i: i + args.bsize], news_train, stocks_train, pos_train, labels_train)
        news = input_dict['news']
        stocks = input_dict['stocks'].cuda()
        sent_lens = input_dict['sent_lens']
        stock_lens = input_dict['stock_lens']
        event = input_dict['pos'].cuda()
        labels = input_dict['labels'].view(-1, 1).cuda()
        output_dict = model(news, stocks, sent_lens, stock_lens, event = event)
        loss_bce = criterion(output_dict['probs'], labels)
        if args.mode == 2:
            loss_crf = output_dict['loss_crf']
            loss = args.lmbd * loss_crf + (1 - args.lmbd) * loss_bce
        else:
            loss = loss_bce
        loss.backward()
        optimizer.step()
        minibatch_iter.set_postfix(loss = loss.item())

def evaluate(curr_epoch, model):
    model.eval()
    optimizer.zero_grad()
    correct_total = 0
    num_total = 0
    minibatch_iter = tqdm.tqdm(range(0, num_valid, args.bsize), desc = f"(Epoch {curr_epoch}) Minibatch")
    for i in minibatch_iter:
        input_dict = batch_sampler(np.arange(i, i + args.bsize), news_train, stocks_train, pos_train, labels_train)
        news = input_dict['news']
        stocks = input_dict['stocks'].cuda()
        sent_lens = input_dict['sent_lens']
        stock_lens = input_dict['stock_lens']
        event = input_dict['pos'].cuda()
        labels = input_dict['labels'].view(-1, 1).cuda()
        output_dict = model(news, stocks, sent_lens, stock_lens, event = event)
        pred = (output_dict['probs'] >= 0.5).to(torch.int)
        num_correct = (pred == labels).sum().cpu().item()
        correct_total += num_correct
        num_total += labels.size(0)
        minibatch_iter.set_postfix(acc = num_correct / labels.size(0))
    print('Acc. on test set: {:.3f}'.format(correct_total / num_total))
    return correct_total / num_total

if not args.evaluate:
    model = model.cuda()
    prev_best = 0.
    for e in range(args.nepochs):
        train_epoch(e, model, criterion, optimizer)
        evaluate(e, model)
        if valid_acc > prev_best:
            print('Saving...')
            state = {'net': net.state_dict(), 'acc': valid_acc, 'epoch': e}
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
else:
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['net'])
    model = model.cuda()
    evaluate(-1, model)
