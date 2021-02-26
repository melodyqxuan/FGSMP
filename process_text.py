# TODOs
# document all vocabs occurred in the corpus
# the vocab_size = vocab_size of corpus + 3, i.e., <sos>, <eos>, <pad>
# batch loader for training examples
# for a batch x = [x_1,...,x_B], sort the length of sentence in this bach by descending order
# pad each x_i with <sos> x_i <eos> <pad> ... <pad> (padding until the length of x_i is equal to the longest in the batch)

import json 
import pandas as pd 
import nltk
import torch
import numpy as np 
import torch.utils.data as data 
#from nltk.parse import CoreNLPParser
#from nltk.parse.corenlp import CoreNLPDependencyParser
import urllib.request
import stanza 

f = open('dummy_news.json') #creates dictionary with key-value pairs of the JSON string

data_f = json.load(f)
data_f = sorted(data_f, key = lambda i: i['pub_time'])
#POS Tagging
#pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype = 'pos')
#list(pos_tagger.tag('Sentences'))

#Dependency Relation
#dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
#parses = list(dep_parser.parse('Sentences'))
#[[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for parse in parses]



titles = []
texts = []
pub_times = []
pos_tag = []
dep_relation = []

pos =  stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
dep = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
for i in data_f:
	if False:
		pass
	else:
		titles.append(i['title'])
		texts.append(i['text'])
		pub_times.append(i['pub_time'])
		doc_pos = pos(i['title'])
		doc_dep = dep(i['title'])
		pos_tag.append(doc_pos.to_dict())
		dep_relation.append(doc_dep.to_dict())
		
		# try:
		# 	pos_tmp = pos_tagger.tag(i['title'].split() + i['text'].split())
		# except:
		# 	print('Error occured at index:', ind)
		# 	continue
		# pos_tag.append(pos_tmp)
		# try:
		# 	dep_tmp=dep_parser.parse(i['title'].split() + i['text'].split())
		# except:
		# 	continue
		# dep_relation.append([[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for parse in dep_tmp])
		# texts.append(i['text'])
		# pub_times.append(i['pub_time'])

news = pd.DataFrame({'title':titles, 'text': texts, 'time':pub_times, 'pos': pos_tag, 'dep': dep_relation})

savefile = './data_text.npz'
np.savez_compressed(savefile, data = data_f, df = news, pos = pos_tag, dep = dep_relation, title = titles, text = texts, time = pub_times)


SOS_token = 0
EOS_token = 1

class BuildVocab:
	"""Build vocabulary"""
	def __init__(self):
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}
		self.n_words = 3

	def addText(self, text):
		for sentence in text:
			self.addSentence(sentence)

	def addSentence(self, sentence):
		for word in nltk.word_tokenize(sentence):
			self.addWord(word)
	def addWord(self, word):
		if not word.isalpha():
			pass
		else:
			if word not in self.word2index:
				self.word2index[word] = self.n_words
				self.word2count[word] = 1
				self.index2word[self.n_words] = word 
				self.n_words += 1
			else:
				self.word2count[word] += 1


def TextToSentences(text):
	"""
	Split texts into sentences 
	"""
	sentences = nltk.sent_tokenize(text)
	return sentences



vocab = BuildVocab()

for i in range(len(data_f)):
	text_title = TextToSentences(titles[i])
	vocab.addText(text_title)
	text_body = TextToSentences(texts[i])
	vocab.addText(text_body)









class Dataset(data.Dataset):
	def __init__(self, path, word2id):
		f = open(path) #creates dictionary with key-value pairs of the JSON string
		data_f = json.load(f)
		self.seq = []
		for i in data_f:
			self.seq.append(i['title'] +'. '+ i['text'])
		self.num_tot_seqs = len(self.seq)
		self.word2id = word2id 

	def __getitem__(self, index):
		"""return one data pair"""
		seq = self.seq[index]
		seq = self.preprocess(seq, self.word2id)
		return seq 

	def __len__(self):
		return self.num_tot_seqs 

	def preprocess(self, sequence, word2id):
		"""convert words to ids"""
		tokens = nltk.tokenize.word_tokenize(sequence.lower())
		sequence = []
		sequence.append(word2id['<SOS>'])
		sequence.extend([word2id[token] for token in tokens if token in word2id])
		sequence.append(word2id['<EOS>'])
		sequence = torch.Tensor(sequence)
		return sequence 

def collate_fn(data):
	"""
	Create mini-batch tensors
	data: 
		- seq:
	returns:
		- padded_seqs: torch tensor of shape(batch_size, padded_length)
		- lengths: list of lengths(batch_size); 
	"""
	def merge(sequences):
		lengths = [len(seq) for seq in sequences]
		padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
		for i, seq in enumerate(sequences):
			end = lengths[i]
			padded_seqs[i, end:] = '<PAD>' 
		return padded_seqs, lengths 


	#sort a list by sequence length (descending order)
	data.sort(key = lambda x: len(x[0]), reverse = True)
	padded_seqs, lengths = merge(data)

	return padded_seqs, lengths



def getLoader(path, word2id, batch_size = 10):
	"""
	Returns data loader for custom dataset
	"""

	dataset = Dataset(path, word2id)
	data_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, 
		shuffle = True, collate_fn = collate_fn)

	return data_loader







##TOPIX dictionary 
trigger = { 
0:['announcement', 'announce'],
1:['court', 'litigation', 'lawsuit'],
2:['appint', 'name' + 'as'],
3:['recall'],
4:['buy'],
5:['alliance', 'partnership'],
6:['order', 'deal', 'trade'],
7:['demand', 'supply'],
8:['take' + 'charge', 'invest', 'investment'],
9:['sale'],
10:['share issuance'],
11:['div', 'dividend'],
12:['fund', 'funding'],
13:['ipo'],
14:['jv', 'joint venture'],
15:['acquisition', 'merge', 'acquire'],
16:['buy' + 'back', 'share repurchase'],
17:['share issuance'],
18:['stock split'],
19:['product', 'production', 'project'],
20:['profit', 'group result', 'earnings', 'parent result', 'financial result'],
21:['earnings adjust', 'profit adjust'],
22:['group forecast', 'parent forecast'],
23:['solar power', 'plant'],
24:['social', 'society'],
25:['gove', 'government'],
26:['shares down', 'share up', 'trade hult'],
27:['holder', 'investor'],
28:['oil', 'coal', 'gasstel', 'fuel', 'crude', 'copper'],
29:['u.s.', 'us', 'american', 'china', 'uk', 'eu', 'europe', 'asia', 'asian', 'indonesia','india','australia'],
30:['target price', 'rating', 'rase' + 'price', 'switch' + 'to', 'scah' + 'to'],
31:['moody']}
















