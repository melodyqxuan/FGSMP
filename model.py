import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SentBiLSTM(nn.Module):
	def __init__(self, vocab_size, embed_dim = 256, hidden_dim = 256, num_layers = 1, bidirectional = True):
		'''A minimal implementation of SentBiLSTM, in which we assume that the vocabulary of size
		   vocab_size covers all unique words occurred in our training corpus
		Parameters
			vocab_size: size of vocabulary
			embed_dim: embedding dimension
			hidden_dim: hidden dimension for LSTM
			num_layers: number of hidden layers for LSTM
			bidirectional: whether the LSTM is bi-directional
		'''
		super(SentBiLSTM, self).__init__()
		self.vocab_size = vocab_size
		self.embed_dim = embed_dim
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		
		# initialize model parameters
		# we provide a skeleton implementation with minimal hyperparameters
		self.embedding = nn.Embedding(vocab_size, embed_dim)
		self.encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, \
							   num_layers=num_layers, bidirectional=bidirectional)
		
	def forward(self, x, lens):
		'''
		Parameters:
			x: a batch of padded input sequence of size (bsize, max_len); where max_len = max(lens)
			lens: an array of length bsize, where lens[i] corresponds to the actual sentence length of x[i]
		'''
		out = self.embedding(x.t()) # (len, bsize, embed_dim)
		out = pack_padded_sequence(out, lens)
		out, _ = self.encoder(out)
		out, _ = pad_packed_sequence(out)
		return out

class StockBiLSTM(nn.Module):
	def __init__(self, stock_dim, hidden_dim = 256, num_layers = 1, bidirectional = True):
		raise NotImplementedError
		super(StockBiLSTM, self).__init__()
		self.stock_dim = stock_dim
		self.num_layers = num_layers 
		self.bidirectional = bidirectional 

		#initialize model parameters 
		self.encoder = nn.LSTM(input_size = stock_dim, hidden_size = hidden_dim, \
								num_layers = num_layers, bidirectional = bidirectional)

	def forward(self, x, lens):
		out , _ = self.encoder(out)
		return out 

class TensorFusion(nn.Module):
	def __init__(self, input_dim, output_dim, activation = 'ReLU', bias = True):
		'''A specialized Linear layer for tensorfusion
		'''
		super(TensorFusion, self).__init__()
		self.input_dim = 4 * input_dim
		self.output_dim = output_dim
		self.fc = nn.Linear(self.input_dim, self.output_dim, bias = bias)
		self.activation = getattr(nn, activation)()

	def forward(self, x, y):
		input = torch.cat((x, y, x - y, x * y), dim=2)
		out = self.activation(self.fc(input))
		return out

class CoAttn(nn.Module):
	def __init__(self, dim_x, dim_y):
		super(CoAttn, self).__init__()
		self.dim_x = dim_x
		self.dim_y = dim_y
		self.activation = nn.ReLU()
		self.M = Variable(nn.init.xavier_uniform_(torch.empty(dim_x, dim_y)))
		if torch.cuda.is_available():
			self.M = self.M.cuda()

	def forward(self, x, y, mask_x = None, mask_y = None):
		'''Perform Co-Attention for input x (bsize, L, dim_x) and y (bsize, T, dim_y)
		   where L and T are sequence length for x and y respectively
		Parameters:
			x: a batch of shape (bsize, L, dim_x), assumed to be the sentence encoder output
			y: a batch of shape (bsize, T, dim_y), assumed to be the stock encoder output
			mask_x: a mask of shape (bsize, L), where True indicates that this position will be masked
			mask_y: a mask of shape (bsize, T), defined similarly to mask_x
		'''
		scores = torch.matmul(torch.matmul(x, self.M), y.transpose(-2, -1))
		scores = self.activation(scores)
		if mask_x is not None and mask_y is not None:
			mask_x = mask_x.to(torch.bool).unsqueeze(2)
			mask_y = mask_y.to(torch.bool).unsqueeze(1)
			mask = mask_x + mask_y # mask of shape (bsize, L, T)
		scores = scores.masked_fill_(mask, -1e9)

		A = F.softmax(scores, dim=2) # attn mapping over stocks
		B = F.softmax(scores, dim=1) # attn mapping over words
		#print(A.size(), B.size())

		C_x = torch.bmm(A, y)
		C_y = torch.bmm(B.transpose(1, 2), x)
		return C_x, C_y

class GatedSum(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(GatedSum, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim

		self.enc_prev = nn.Linear(input_dim, output_dim, bias=False)
		self.enc_attn = nn.Linear(input_dim, output_dim, bias=False)

	def forward(self, x, y):
		z = torch.sigmoid(self.enc_prev(x) + self.enc_attn(y))
		out = z * y + (1 - z) * x
		return out

class CRF(nn.Module):
	def __init__(self, input_dim, event_size, normalize = True):
		'''Implements BiLSTM-CRF
		Parameters:
			input_dim: input_dimension (i.e. 2 * hidden_dim)
			event_size: event dimension
			normalize: whether to normalize the loss w.r.t sequence length
		'''
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
			x: (L, bsize, dim)
			tags: target tags (bsize, L)
			mask: (bsize, L)
		'''
		bsize, L = tags.size()
		mask_ = 1. - mask.to(torch.float)
		x = x.transpose(0, 1)
		emit_score = F.softmax(self.fc(x), dim=2) # (bsize, L, event_size)
		score = torch.gather(emit_score, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)
		score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]
		total_score = (score * mask_).sum(dim=1)  # (bsize,)

		d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (bsize, 1, event_size)
		for i in range(1, L):
			n_unfinished = mask_[:, i].sum().cpu().to(torch.int).item()
			d_uf = d[:n_unfinished]
			emit_and_transition = emit_score[:n_unfinished, i].unsqueeze(dim=1) + self.transition
			log_sum = d_uf.transpose(1, 2) + emit_and_transition
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
			x: (L, bsize, dim)
			lens: an array of sentence lengths of length bsize
			mask: (bsize, L)
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
	def __init__(self, input_dim, output_dim = 1, pooling = 'MaxPool1d'):
		super(Classifier, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.fc = nn.Linear(input_dim, output_dim)
		self.pooling = pooling

	def forward(self, x, y, mask_x, mask_y):
		'''First perform a global pooling operation over x and y respectively
		   then compute the logit
		Parameters
			x: (bsize, L, dim)
			y: (bsize, T, dim)
		'''
		x = x.transpose(1, 2)
		x_in = self.pool(x, mask_x)
		y = y.transpose(1, 2)
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

class SSPM(nn.Module):

	def __init__(self, vocab_size, word_embed_dim, hidden_dim, event_size, num_heads):
		assert (hidden_dim * 2) % num_heads == 0, 'Attention hidden dimension is not divisible by the number of attention heads'
		super(SSPM, self).__init__()
		
		self.sent_enc = SentBiLSTM(vocab_size=vocab_size, embed_dim=word_embed_dim, hidden_dim=hidden_dim)
		self.sent_attn = nn.MultiheadAttention(2 * hidden_dim, num_heads=num_heads)
		self.event_enc = nn.Embedding(event_size, 2 * hidden_dim)

		# TODO: implement StockBiLSTM, currently a SentBiLSTM is used instead
		self.stock_enc = SentBiLSTM(vocab_size=vocab_size, embed_dim=word_embed_dim, hidden_dim=hidden_dim)
		self.stock_attn = nn.MultiheadAttention(2 * hidden_dim, num_heads=num_heads)
		# TODO end

		self.fuse = TensorFusion(2 * hidden_dim, 2 * hidden_dim)
		self.coattn = CoAttn(dim_x=2 * hidden_dim, dim_y=2 * hidden_dim)

		self.sent_gate = GatedSum(2 * hidden_dim, 2 * hidden_dim)
		self.stock_gate = GatedSum(2 * hidden_dim, 2 * hidden_dim)

		self.classify = Classifier(4 * hidden_dim)
		self.gpu = torch.cuda.is_available()

	def compute_mask(self, x, x_lens):
		mask = torch.ones(len(x_lens), x_lens[0])
		for i, l in enumerate(x_lens):
			mask[i, :l] = 0
		mask = mask.to(torch.bool)
		if self.gpu:
			mask = mask.cuda()
		return mask

	def forward(self, sent, event, stock, sent_lens, stock_lens):
		event = event.t()
		if self.gpu:
			sent = sent.cuda()
			event = event.cuda()
			stock = stock.cuda()

		sent_mask = self.compute_mask(sent, sent_lens)
		stock_mask = self.compute_mask(stock, stock_lens)

		sent_out = self.sent_enc(sent, sent_lens)
		sent_out, _ = self.sent_attn(sent_out, sent_out, sent_out, key_padding_mask=sent_mask)
		event_out = self.event_enc(event)
		sent_out = self.fuse(sent_out, event_out)

		stock_out = self.stock_enc(stock, stock_lens)
		stock_out, _ = self.stock_attn(stock_out, stock_out, stock_out, key_padding_mask=stock_mask)

		sent_out = sent_out.transpose(0, 1).contiguous()
		stock_out = stock_out.transpose(0, 1).contiguous()

		sent_coattn, stock_coattn = self.coattn(sent_out, stock_out, sent_mask, stock_mask)
		sent_g = self.sent_gate(sent_out, sent_coattn)
		stock_g = self.stock_gate(stock_out, stock_coattn)

		logits = self.classify(sent_g, stock_g, sent_mask, stock_mask)
		probs = nn.Sigmoid()(logits)
		return probs

class MSSPM(nn.Module):

	def __init__(self, vocab_size, word_embed_dim, hidden_dim, event_size, num_heads):
		assert (hidden_dim * 2) % num_heads == 0, 'Attention hidden dimension is not divisible by the number of attention heads'
		super(MSSPM, self).__init__()
		
		self.sent_enc = SentBiLSTM(vocab_size=vocab_size, embed_dim=word_embed_dim, hidden_dim=hidden_dim)
		self.sent_attn = nn.MultiheadAttention(2 * hidden_dim, num_heads=num_heads)
		self.event_crf = CRF(2 * hidden_dim, event_size)
		self.event_enc = nn.Embedding(event_size, 2 * hidden_dim)

		# TODO: implement StockBiLSTM, currently a SentBiLSTM is used instead
		self.stock_enc = StockBiLSTM(stock_dim=stock_dim, hidden_dim=hidden_dim)
		self.stock_attn = nn.MultiheadAttention(2 * hidden_dim, num_heads=num_heads)
		# TODO end

		self.fuse = TensorFusion(2 * hidden_dim, 2 * hidden_dim)
		self.coattn = CoAttn(dim_x=2 * hidden_dim, dim_y=2 * hidden_dim)

		self.sent_gate = GatedSum(2 * hidden_dim, 2 * hidden_dim)
		self.stock_gate = GatedSum(2 * hidden_dim, 2 * hidden_dim)

		self.classify = Classifier(4 * hidden_dim)
		self.gpu = torch.cuda.is_available()

	def compute_mask(self, x, x_lens):
		mask = torch.ones(len(x_lens), x_lens[0])
		for i, l in enumerate(x_lens):
			mask[i, :l] = 0
		mask = mask.to(torch.bool)
		if self.gpu:
			mask = mask.cuda()
		return mask

	def forward(self, sent, event, stock, sent_lens, stock_lens):
		if self.gpu:
			sent = sent.cuda()
			event = event.cuda()
			stock = stock.cuda()

		sent_mask = self.compute_mask(sent, sent_lens)
		stock_mask = self.compute_mask(stock, stock_lens)

		sent_out = self.sent_enc(sent, sent_lens)
		sent_out, _ = self.sent_attn(sent_out, sent_out, sent_out, key_padding_mask=sent_mask)

		loss_crf, _ = self.event_crf(sent_out, event, sent_mask)
		event_out = self.event_enc(event.t())
		sent_out = self.fuse(sent_out, event_out)

		stock_out = self.stock_enc(stock, stock_lens)
		stock_out, _ = self.stock_attn(stock_out, stock_out, stock_out, key_padding_mask=stock_mask)

		sent_out = sent_out.transpose(0, 1).contiguous()
		stock_out = stock_out.transpose(0, 1).contiguous()

		sent_coattn, stock_coattn = self.coattn(sent_out, stock_out, sent_mask, stock_mask)
		sent_g = self.sent_gate(sent_out, sent_coattn)
		stock_g = self.stock_gate(stock_out, stock_coattn)

		logits = self.classify(sent_g, stock_g, sent_mask, stock_mask)
		probs = nn.Sigmoid()(logits)
		return probs, loss_crf


if __name__ == '__main__':
	word_emb_dim = 8
	hidden_dim = 10
	vocab_size = 16
	event_size = 16
	num_heads = 4
	eta = 0.5
	criterion = nn.BCELoss()
	#model = SSPM(vocab_size, word_emb_dim, hidden_dim, event_size, num_heads)
	model = MSSPM(vocab_size, word_emb_dim, hidden_dim, event_size, num_heads)
	model.cuda()


	torch.manual_seed(619)
	word_lens = [9, 6, 5, 3, 2]
	stock_lens = [12, 12, 12, 12, 12]
	x = [torch.cat((torch.randint(low=1, high=8, size=(i,)), torch.zeros(9 - i))).long() for i in word_lens]
	x = torch.stack(x)
	e = [torch.cat((torch.randint(low=1, high=8, size=(i,)), torch.zeros(9 - i))).long() for i in word_lens]
	e = torch.stack(e)
	s = [torch.cat((torch.randint(low=1, high=8, size=(i,)), torch.zeros(12 - i))).long() for i in stock_lens]
	s = torch.stack(s)
	y = torch.Tensor([0, 1, 1, 0, 0]).view(5, 1).cuda()

	probs, loss_crf = model(x, e, s, word_lens, stock_lens)
	optimizer = optim.Adam(model.parameters(), lr=1e-2)

	loss_bce = criterion(probs, y)
	loss = eta * loss_crf + (1 - eta) * loss_bce
	loss.backward()

	#sent_enc = SentBiLSTM(vocab_size=vocab_size, embed_dim=word_emb_dim, hidden_dim=hidden_dim)
	#sent_attn = nn.MultiheadAttention(2 * hidden_dim, num_heads = num_heads)
	#event_enc = nn.Embedding(event_size, 2 * hidden_dim)
	#fuse = TensorFusion(2 * hidden_dim, 2 * hidden_dim)
	#merge = CoAttn(dim_x=2 * hidden_dim, dim_y=2 * hidden_dim)
	#word_gate = GatedSum(2 * hidden_dim, 2 * hidden_dim)
	#stock_gate = GatedSum(2 * hidden_dim, 2 * hidden_dim)
	#classify = Classifier(4 * hidden_dim)
	

	# sentence encoding
	#sent_out = sent_enc(x, word_lens)
	#sent_mask = torch.ones(len(word_lens), word_lens[0])
	#for i, l in enumerate(word_lens):
	#	sent_mask[i,:l] = 0
	#sent_mask = sent_mask.to(torch.bool)
	#sent_out, _ = sent_attn(sent_out, sent_out, sent_out, key_padding_mask=sent_mask)

	# dummy stock encoding
	

	#stock_enc = SentBiLSTM(vocab_size=vocab_size, embed_dim=word_emb_dim, hidden_dim=hidden_dim)
	#stock_attn = nn.MultiheadAttention(2 * hidden_dim, num_heads = num_heads)
	#stock_out = stock_enc(s, stock_lens)
	#stock_mask = torch.ones(len(stock_lens), stock_lens[0])
	#for i, l in enumerate(stock_lens):
	#	stock_mask[i, :l] = 0
	#stock_mask = stock_mask.to(torch.bool)
	#stock_out, _ = stock_attn(stock_out, stock_out, stock_out, key_padding_mask=stock_mask)

	# event encoding
	#event_out = event_enc(e.t())

	# tensor fusion
	#fuse_out = fuse(sent_out, event_out)

	# co-attention
	#fuse_out = fuse_out.transpose(0, 1).contiguous()
	#stock_out = stock_out.transpose(0, 1).contiguous()
	#Cx, Cy = merge(fuse_out, stock_out, sent_mask, stock_mask)
	#word_gs = word_gate(fuse_out, Cx)
	#stock_gs = stock_gate(stock_out, Cy)

	#out = classify(word_gs, stock_gs, sent_mask, stock_mask)
	#probs = nn.Sigmoid()(out)
	
	


	





