import os, time, argparse
import tqdm
import numpy as np

import torch
import torch.nn as nn
from model import MSSPM
from data_utils import build_dataset, batch_sampler

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
args = parser.parse_args()

news_train, stocks_train, pos_train, labels_train = build_dataset(args.train_root)
news_valid, stocks_valid, pos_valid, labels_valid = build_dataset(args.valid_root)
num_train = len(news_train)
num_valid = len(news_valid)

event_size = 38
model = MSSPM(args.hidden_dim, args.sent_encoder_layers, args.stock_encoder_layers, \
              args.num_heads, args.dropout, args.mode, event_size = event_size)
num_params = sum(p.numel() for p in model.parameters())
print('Number of Parameters: {}'.format(num_params))
model = model.cuda()
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

for e in range(args.nepochs):
    train_epoch(e, model, criterion, optimizer)
    evaluate(e, model)
