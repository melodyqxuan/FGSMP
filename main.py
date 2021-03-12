import os, time, argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from model import MSSPM
from data_utils import build_dataset, batch_sampler

parser = argparse.ArgumentParser(description='TBD')
parser.add_argument('--train-root', default='train')
parser.add_argument('--valid-root', default='valid')
parser.add_argument('--mode', default=0, type=int) #0, 1, 2
parser.add_argument('--hidden-dim', default=256, type=int)
parser.add_argument('--sent-encoder-layers', default=3)
parser.add_argument('--stock-encoder-layers', default=3)
parser.add_argument('--num-heads', default=4)
parser.add_argument('--dropout', default=0.2, type=float)

parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--weight_decay', default=0.)
parser.add_argument('--bsize', default=32, type=int)
parser.add_argument('--nepochs', default=50)
args = parser.parse_args()


news_train, stocks_train, pos_train, labels_train = build_dataset(args.train_root)
news_valid, stocks_valid, pos_valid, labels_valid = build_dataset(args.valid_root)
num_train = len(news_train)

event_size = 38
model = MSSPM(args.hidden_dim, args.sent_encoder_layers, args.stock_encoder_layers, \
              args.num_heads, args.dropout, args.mode, event_size = event_size)
num_params = sum(p.numel() for p in model.parameters())
print('Number of Parameters: {}'.format(num_params))
model = model.cuda()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

def train_epoch(curr_epoch, model, criterion, optimizer):
    ind = np.random.permutation(num_train)
    with tqdm(total = num_train // args.bsize) as pbar:
        for i in range(0, num_train, args.bsize):
            optimizer.zero_grad()
            input_dict = batch_sampler(ind[i: i + args.bsize], news_train, stocks_train, pos_train, labels_train)
            news = input_dict['news']
            stocks = input_dict['stocks'].cuda()
            sent_lens = input_dict['sent_lens']
            stock_lens = input_dict['stock_lens']
            event = input_dict['pos'].cuda()
            labels = input_dict['labels'].view(-1, 1).cuda()
            output_dict = model(news, stocks, sent_lens, stock_lens, event = event)
            #TODO: mode = 2
            loss_bce = criterion(output_dict['probs'], labels)
            loss_bce.backward()
            print(loss_bce.detach().cpu().item())
            optimizer.step()
            pbar.update()


for e in range(args.nepochs):
    train_epoch(e, model, criterion, optimizer)
