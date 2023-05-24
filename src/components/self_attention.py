import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time
from torch.autograd import Variable
import ipdb as pdb
from src.components.utils import *


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # pdb.set_trace()
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, bias=True,
                 freeze_q=False, freeze_k=False,
                 freeze_v=False, zero_k=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        self.bias = bias
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model, bias=bias), 4)
        if freeze_q:
            self.linears[0].requires_grad_(False)
        if freeze_k:
            self.linears[1].requires_grad_(False)
        if freeze_v:
            self.linears[2].requires_grad_(False)
        if zero_k:
            self.null_linear_layer(self.linears[1])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def null_linear_layer(self, ln):
        with torch.no_grad():
            ln.weight.fill_(0.0)
            if self.bias:
                ln.bias.fill_(0.0)
        ln.requires_grad_(False)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(0).unsqueeze(0)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# RelativeAttention based on Jake Tae's implementation: https://jaketae.github.io/study/relative-positional-encoding/

# class RelativeGlobalAttention(nn.Module):
#     def __init__(self, num_heads, d_model, max_len=1024, dropout=0.1):
#         super().__init__()
#         d_head, remainder = divmod(d_model, num_heads)
#         if remainder:
#             raise ValueError(
#                 "incompatible `d_model` and `num_heads`"
#             )
#         self.max_len = max_len
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.key = nn.Linear(d_model, d_model)
#         self.value = nn.Linear(d_model, d_model)
#         self.query = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.Er = nn.Parameter(torch.randn(max_len, d_head))
#         self.register_buffer(
#             "mask",
#             torch.tril(torch.ones(max_len, max_len))
#             .unsqueeze(0).unsqueeze(0)
#         )
#
#     # self.mask.shape = (1, 1, max_len, max_len)
#
#     def forward(self, x):
#         # x.shape == (batch_size, seq_len, d_model)
#         batch_size, seq_len, _ = x.shape
#
#         if seq_len > self.max_len:
#             raise ValueError(
#                 "sequence length exceeds model capacity"
#             )
#
#         k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
#         # k_t.shape = (batch_size, num_heads, d_head, seq_len)
#         v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
#         q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
#         # shape = (batch_size, num_heads, seq_len, d_head)
#
#         start = self.max_len - seq_len
#         Er_t = self.Er[start:, :].transpose(0, 1)
#         # Er_t.shape = (d_head, seq_len)
#         QEr = torch.matmul(q, Er_t)
#         # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
#         Srel = self.skew(QEr)
#         # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
#
#         QK_t = torch.matmul(q, k_t)
#         # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
#         attn = (QK_t + Srel) / math.sqrt(q.size(-1))
#         mask = self.mask[:, :, :seq_len, :seq_len]
#         # mask.shape = (1, 1, seq_len, seq_len)
#         attn = attn.masked_fill(mask == 0, float("-inf"))
#         # attn.shape = (batch_size, num_heads, seq_len, seq_len)
#         attn = F.softmax(attn, dim=-1)
#         out = torch.matmul(attn, v)
#         # out.shape = (batch_size, num_heads, seq_len, d_head)
#         out = out.transpose(1, 2)
#         # out.shape == (batch_size, seq_len, num_heads, d_head)
#         out = out.reshape(batch_size, seq_len, -1)
#         # out.shape == (batch_size, seq_len, d_model)
#         return self.dropout(out)
#
#     def skew(self, QEr):
#         # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
#         padded = F.pad(QEr, (1, 0))
#         # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
#         batch_size, num_heads, num_rows, num_cols = padded.shape
#         reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
#         # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
#         Srel = reshaped[:, :, 1:, :]
#         # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
#         return Srel
