# __author__ = 'zhuqing'
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
	def __init__(self,d_in,d_hidden,dropout=0.1):
		super().__init__()
		self.w_1 = nn.Conv1d(d_in,d_hidden,1)
		self.w_2 = nn.Conv1d(d_hidden,d_in,1)
		self.layer_norm = nn.LayerNorm(d_in)
		self.dropout = nn.Dropout(dropout)

	def forward(self,x):
		residual = x
		output = x.transpose(1,2)
		output = self.w_1(output)
		output = F.relu(output)
		output = self.w_2(output)
		output = self.dropout(output)
		output = self.layer_norm(output+residual)
		return output