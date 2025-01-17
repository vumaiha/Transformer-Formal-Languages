import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb as pdb

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
	r"""Inject some information about the relative or absolute position of the tokens
		in the sequence. The positional encodings have the same dimension as
		the embeddings, so that the two can be summed. Here, we use sine and cosine
		functions of different frequencies.
	.. math::
		\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
		\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
		\text{where pos is the word position and i is the embed idx)
	Args:
		d_model: the embed dim (required).
		dropout: the dropout value (default=0.1).
		max_len: the max. length of the incoming sequence (default=5000).
	Examples:
		>>> pos_encoder = PositionalEncoding(d_model)
	"""

	def __init__(self, d_model, dropout=0.1, max_period = 10000.0, max_len=500):
		super(PositionalEncoding, self).__init__()
		odd_flag=False
		if int(d_model%2) !=0:
			odd_flag=True
		self.dropout = nn.Dropout(p=dropout)
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(max_period) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		if odd_flag:
			pe[:, 1::2] = torch.cos(position * div_term[:-1])
		else:
			pe[:, 1::2] = torch.cos(position * div_term)
		#print(pe)
		pe = pe.unsqueeze(0).transpose(0, 1)
		#print(pe)
		self.register_buffer('pe', pe)

	def forward(self, x):
		r"""Inputs of forward function
		Args:
			x: the sequence fed to the positional encoder model (required).
		Shape:
			x: [sequence length, batch size, embed dim]
			output: [sequence length, batch size, embed dim]
		Examples:
			>>> output = pos_encoder(x)
		"""

		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)

class CosineNpiPositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(CosineNpiPositionalEncoding, self).__init__()
		odd_flag=False
		if int(d_model%2) !=0:
			odd_flag=True
		self.dropout = nn.Dropout(p=dropout)
		pe = torch.ones(max_len, d_model)
		#print(max_len)
		#print(d_model)
		for i in range(max_len):
			pe[i] = pe[i] * math.cos(i * math.pi)
		#print(pe)
		pe = pe.unsqueeze(0).transpose(0, 1)
		#print(pe)
		self.register_buffer('pe', pe)

	def forward(self, x):

		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)

class PeriodNPositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout=0.1, max_len=500, periodicity=3): #changed max_len to 500 to match with original absolute encoding
		super(PeriodNPositionalEncoding, self).__init__()
		odd_flag=False
		if int(d_model%2) !=0:
			odd_flag=True
		self.dropout = nn.Dropout(p=dropout)
		pe = torch.ones(max_len, d_model)
		#print(max_len)
		#print(d_model)
		#print(pe)
		for i in range(max_len):
			pe[i] = pe[i] * (i%periodicity)/(periodicity-1)
		#print(pe)
		pe = pe.unsqueeze(0).transpose(0, 1)
		#print(pe)
		self.register_buffer('pe', pe)

	def forward(self, x):

		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout=0.1, max_len=5000, init_range = 0.1):
		super(LearnablePositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		pos_embeds = torch.FloatTensor(max_len, 1, d_model).uniform_(-init_range, init_range)
		pe = nn.Parameter(pos_embeds, requires_grad = True)
		self.pe = pe

	def forward(self, x):
		#pdb.set_trace()
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)
