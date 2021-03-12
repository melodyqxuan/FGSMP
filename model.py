import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.modules.elmo import Elmo, batch_to_ids

class BiLSTM(nn.Module):
    '''A minimal implementation of Bi-directional LSTM
    Parameters:
        input_dim: LSTM input dimension
        hidden_dim: LSTM hidden dimension
        num_layers: number of LSTM hidden layers
        bidirectional: whether the hidden representations are encoded bi-directionally
        dropout: LSTM dropout probability
    '''
    def __init__(self, input_dim = 1024, hidden_dim = 256, num_layers = 3, bidirectional = True, dropout = 0.2):
        super(BiLSTM, self).__init__()
        self.encoder = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, dropout = dropout, \
                               num_layers = num_layers, bidirectional = bidirectional)

    def forward(self, x, lens):
        '''
        Parameters:
            x: a batch of padded input sequence of size (bsize, max_len, input_dim)
            lens: an array of length bsize, where lens[i] corresponds to the actual sentence length of x[i]
        '''
        x = x.transpose(0, 1).contiguous()
        out = pack_padded_sequence(x, lens, enforce_sorted = False)
        out, _ = self.encoder(out)
        out, _ = pad_packed_sequence(out)
        return out

class TensorFusion(nn.Module):
    '''Implements a simple TensorFusion module to aggregate event label embedding and stock
       news sequence representations. Given two input vectors x, y of size (*, input_dim),
       this module computes activation([x, y, x - y, x * y]W + b), where * is performed element wise
    Parameters:
        input_dim: fusion input dimension
        output_dim: fusion output dimension
        activation: activation function. Must be defined within torch.nn
        bias: whether to add bias vector in the linear layer
    '''
    def __init__(self, input_dim, output_dim, activation = 'ReLU', bias = True):
        super(TensorFusion, self).__init__()
        fc = nn.Linear(4 * input_dim, output_dim, bias = bias)
        activation = getattr(nn, activation)()
        self.encoder = nn.Sequential(fc, activation)

    def forward(self, x, y):
        input = torch.cat((x, y, x - y, x * y), dim = 2)
        out = self.encoder(input)
        return out

class CoAttention(nn.Module):
    '''Implements the CoAttention module as described in the paper. We firt define a learnable
       bilinear matrix variable BiLinearM for computing attention outputs
    Parameters:
        dim_x: dimension associated with input vector x
        dim_y: dimension associated with input vector y
        activation: activation function. Must be defined within torch.nn
    '''
    def __init__(self, dim_x, dim_y, activation = 'ReLU'):
        super(CoAttention, self).__init__()
        self.activation = getattr(nn, activation)()
        self.BiLinearM = Variable(nn.init.xavier_uniform_(torch.empty(dim_x, dim_y)))
        if torch.cuda.is_available():
            self.BiLinearM = self.BiLinearM.cuda()

    def forward(self, x, y, mask_x = None, mask_y = None):
        '''Perform co-attention for input x of shape (bsize, L, dim_x) and y of shape (bsize, T, dim_y)
           where L and T are sequence lengths for x and y respectively
        Parameters:
            x, y: tensors of shapes (bsize, L, dim_x) and (bsize, T, dim_y), representing sentence and
                  stock encoder outputs respectively
            mask_x: a binary mask of shape (bsize, L), where a True entry indicates that the corresponding will be masked
            mask_y: defined similarly to mask_x, but for stock sequence y
        '''
        scores = torch.matmul(torch.matmul(x, self.BiLinearM), y.transpose(-2, -1).contiguous())
        scores = self.activation(scores)

        '''
        broadcast a (bsize, L, T)-dimensional boolean mask that assigns True
        whenever either mask_x or mask_y assigns True (i.e. equivalent to OR gate)
        '''
        if mask_x is not None and mask_y is not None:
            mask_x = mask_x.to(torch.bool).unsqueeze(2)
            mask_y = mask_y.to(torch.bool).unsqueeze(1)
            mask = mask_x + mask_y # mask of shape (bsize, L, T)

        scores = scores.masked_fill_(mask, -1e4)
        A = F.softmax(scores, dim = 2) # softmax attention mapping over stocks
        B = F.softmax(scores, dim = 1) # softmax attention mapping over sentence
        C_x = torch.bmm(A, y)
        C_y = torch.bmm(B.transpose(1, 2).contiguous(), x)
        return C_x, C_y

class GatedSum(nn.Module):
    '''Implements a GatedSum module to aggregate pre- and post-attention representations
       This module is created separately for sentence and stock representations
    '''
    def __init__(self, input_dim, output_dim, bias = False):
        super(GatedSum, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.enc_prev = nn.Linear(input_dim, output_dim, bias = bias)
        self.enc_attn = nn.Linear(input_dim, output_dim, bias = bias)

    def forward(self, x, y):
        z = torch.sigmoid(self.enc_prev(x) + self.enc_attn(y))
        out = z * y + (1 - z) * x
        return out

class CRF(nn.Module):
    '''Implements the BiLSTM-fused conditional random field (CRF)
    Parameters:
        input_dim: input_dimension (i.e. 2 * hidden_dim if bi-directional)
        event_size: event_dimension
        normalize: whether to normalize CRF score w.r.t sequence length. Normalization is
                   recommended by the authors to produce loss functions at the same scale
    '''
    def __init__(self, input_dim, event_size, normalize = True):
        super(CRF, self).__init__()
        self.input_dim = input_dim
        self.event_size = event_size
        self.fc = nn.Linear(input_dim, event_size)
        self.transition = nn.Parameter(torch.randn(event_size, event_size))
        self.normalize = normalize

    def forward(self, x, tags, mask):
        '''First perform a linear transformation to produce the emit scores
           then compute CRF objective for downstream learning
        Parameters:
            x: (L, bsize, dim). Input representations
            tags: target tags (bsize, L). Target event (i.e. POS) labels
            mask: (bsize, L). Boolean mask to filter out the paddings
        '''
        bsize, L = tags.size()
        mask_ = 1. - mask.to(torch.float)
        x = x.transpose(0, 1).contiguous()
        emit_score = F.softmax(self.fc(x), dim=2) # (bsize, L, event_size)
        score = torch.gather(emit_score, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)
        score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]
        total_score = (score * mask_).sum(dim=1)  # (bsize,)

        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (bsize, 1, event_size)
        for i in range(1, L):
            n_unfinished = mask_[:, i].sum().cpu().to(torch.int).item()
            d_uf = d[:n_unfinished]
            emit_and_transition = emit_score[:n_unfinished, i].unsqueeze(dim=1) + self.transition
            log_sum = d_uf.transpose(1, 2).contiguous() + emit_and_transition
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)
            log_sum = log_sum - max_v  # shape: (uf, K, K)
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)
            d = torch.cat((d_uf, d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)
        max_d = d.max(dim=-1)[0]
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)
        llk = total_score - d
        loss = -llk
        if self.normalize:
            for i in range(len(loss)):
                loss[i] /= sum(mask_[i])
        loss = loss.mean()
        return loss, tags

    def predict(self, x, lens, mask):
        '''Perform event label prediction based on input representation x
        Parameters:
            x: (L, bsize, dim). Input representations
            lens: an array of sentence lengths of length bsize
            mask: (bsize, L). Boolean mask to filter out the paddings
        '''
        x = x.transpose(0, 1)
        bsize = mask.size(0)
        emit_score = F.softmax(self.fc(x), dim=2)
        tags = [[[i] for i in range(self.event_size)]] * bsize
        mask_ = 1. - mask.to(torch.float)

        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, sen_lengths[0]):
            n_unfinished = mask_[:, i].sum()
            d_uf = d[: n_unfinished]
            emit_and_transition = self.transition + emit_score[: n_unfinished, i].unsqueeze(dim=1)
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()
            tags[: n_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
            d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)
        _, max_idx = torch.max(d, dim=1)
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]
        return tags

class Classifier(nn.Module):
    '''Implements the final classification module that takes a variable-sized input that depends on time-scale,
       and computes a scalar probability for each sample. Although not mentioned in the paper, we use a global
       pooling operation to reduce variable length inputs to a fixed-dimensional representation
    Parameters:
        input_dim: input dimension (must be same for sentence and stock representations)
        output_dim: output dimen
        pooling: pooling method. Must be defined within torch.nn
    '''
    def __init__(self, input_dim, output_dim = 1, pooling = 'MaxPool1d'):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)
        self.pooling = pooling

    def forward(self, x, y, mask_x, mask_y):
        '''First perform a global pooling operation over x and y respectively
           then compute the logit sigmoid(W[x, y])
        Parameters
            x: (bsize, L, dim)
            y: (bsize, T, dim)
        '''
        x = x.transpose(1, 2).contiguous()
        x_in = self.pool(x, mask_x)
        y = y.transpose(1, 2).contiguous()
        y_in = self.pool(y, mask_y)
        input = torch.cat((x_in, y_in), dim=1)
        out = self.fc(input)
        return out

    def pool(self, x, mask):
        len_seq = mask.size(1)
        res = []
        for i, seq_i in enumerate(x):
            ksize = len_seq - mask[i].sum().cpu().item()
            vec = getattr(nn, self.pooling)(kernel_size=ksize)(seq_i[:,:ksize].unsqueeze(0)).squeeze()
            res.append(vec)
        res = torch.stack(res)
        return res

class MSSPM(nn.Module):
    '''Define the (M)SSPM model that incorporates event label as inputs. Due to limited knowledge on the event
       labels, we provide an additional option that ignores the event label modules (i.e. event embedding and
       TensorFusion) completely
    Parameters:
        hidden_dim: hidden dimension for the LSTM feature extractors
        sent_encoder_layers: number of hidden layers for sentence encoder
        stock_encoder_layers: number of hidden layers for stock encoder
        num_heads: number of self-attention heads
        dropout: dropout probability for BiLSTM
        mode: 0: does not include event label; 1: event label with input embedding; 2: event label CRF
        kwargs: parameters related to event label embeddings
    '''
    def __init__(self, hidden_dim, sent_encoder_layers = 3, stock_encoder_layers = 3, num_heads = 4, \
                 dropout = 0.2, mode = 0, **kwargs):
        assert (hidden_dim * 2) % num_heads == 0, 'Attention hidden dimension is not divisible by the number of attention heads'
        assert mode in (0, 1, 2), 'Current mode not supported'
        super(MSSPM, self).__init__()

        # Load pre-trained ELMo embeddings
        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.sent_embed = Elmo(options_file, weight_file, 1, dropout=0)
        self.sent_encoder = BiLSTM(hidden_dim = hidden_dim, num_layers = sent_encoder_layers, dropout = dropout)
        self.sent_attn = nn.MultiheadAttention(2 * hidden_dim, num_heads = num_heads)
        self.stock_encoder = BiLSTM(input_dim = 120, hidden_dim = hidden_dim, num_layers = stock_encoder_layers, dropout = dropout)
        self.stock_attn = nn.MultiheadAttention(2 * hidden_dim, num_heads = num_heads)

        if mode >= 1:
            if mode == 2: self.event_crf = CRF(2 * hidden_dim, kwargs['event_size'])
            self.event_encoder = nn.Embedding(kwargs['event_size'], 2 * hidden_dim)
            self.fuse = TensorFusion(2 * hidden_dim, 2 * hidden_dim)

        self.co_attn = CoAttention(dim_x = 2 * hidden_dim, dim_y = 2 * hidden_dim)
        self.sent_gate = GatedSum(2 * hidden_dim, 2 * hidden_dim)
        self.stock_gate = GatedSum(2 * hidden_dim, 2 * hidden_dim)
        self.classifier = Classifier(4 * hidden_dim)
        self.gpu = torch.cuda.is_available()
        self.mode = mode

    def compute_mask(self, x, x_lens):
        '''A helper function implemented to generate boolean mask required for attention layers
        '''
        mask = torch.ones(len(x_lens), max(x_lens))
        for i, l in enumerate(x_lens):
            mask[i, :l] = 0
        mask = mask.to(torch.bool)
        if self.gpu:
            mask = mask.cuda()
        return mask

    def forward(self, sent, stock, sent_lens, stock_lens, event = None):
        '''
        Parameters:
            sent: raw sentence input of length bsize
            stock: padded stock input of shape (bsize, T = max_len, 120)
            sent_lens: array of sentence lengths
            stock_lens: array of stock history lengths
            event: padded event labels of size (bsize, L)
        '''
        out = {}
        character_ids = batch_to_ids(sent)
        if self.gpu: character_ids = character_ids.cuda()
        elmo_out = self.sent_embed(character_ids)
        elmo_emb = elmo_out['elmo_representations'][0]
        print('elmo passed')
        # need to negate sent_mask in order to input correct boolean values for key_padding_mask
        sent_mask = ~elmo_out['mask']
        sent_out = self.sent_encoder(elmo_emb, sent_lens)
        sent_out, _ = self.sent_attn(sent_out, sent_out, sent_out, key_padding_mask = sent_mask)
        print('sent self attention passed')

        if self.mode >= 1:
            if self.mode == 2:
                loss_crf, _ = self.event_crf(sent_out, event, sent_mask)
                out['loss_crf'] = loss_crf
            event_out = self.event_encoder(event.t())
            sent_out = self.fuse(sent_out, event_out)
        print('event passed')

        if self.gpu:
            stock = stock.cuda()
        stock_mask = self.compute_mask(stock, stock_lens)
        stock_out = self.stock_encoder(stock, stock_lens)
        stock_out, _ = self.stock_attn(stock_out, stock_out, stock_out, key_padding_mask = stock_mask)
        print('stock passed')

        sent_out = sent_out.transpose(0, 1).contiguous()
        stock_out = stock_out.transpose(0, 1).contiguous()
        sent_coattn, stock_coattn = self.co_attn(sent_out, stock_out, sent_mask, stock_mask)
        sent_g = self.sent_gate(sent_out, sent_coattn)
        stock_g = self.stock_gate(stock_out, stock_coattn)

        logits = self.classifier(sent_g, stock_g, sent_mask, stock_mask)
        probs = nn.Sigmoid()(logits)
        out['probs'] = probs
        return out
