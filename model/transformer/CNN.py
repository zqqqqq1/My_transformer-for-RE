import torch
import torch.nn as nn
import numpy as np
from model.transformer import utils
class CNN(nn.Module):
	def __init__(
		self,len_max_seq,k_size,d_pos,in_ch,out_ch,num_class):
		super().__init__()
		self.relative_position = nn.Embedding(
			len_max_seq*2+1,d_pos,padding_idx=utils.PAD)
		self.cnn1ds = []
		for i in range(len(k_size)):
			cnn1d = nn.Sequential(
				nn.Conv1d(in_channels=in_ch,out_channels=out_ch,kernel_size = k_size[i]),
					nn.ReLU(),
					nn.MaxPool1d(kernel_size=len_max_seq - k_size[i]+1))
			self.cnn1ds.append(cnn1d)
		self.linear = nn.Linear(in_features=out_ch*len(k_size),out_features=num_class)
	def forward(self,dec_output,pos1s,pos2s):
		try:
			en1_positions = self.relative_position(pos1s)
		except Exception as e:
			print(e)
		else:
			pass
		finally:
			pass
		en2_positions = self.relative_position(pos2)
		dec_output = torch.cat([dec_output,en1_positions , en2_positions],dim=2)
		dec_output = dec_output.permute(0,2,1)

		outs = [conv(dec_output) for conv in self.conn1ds]
		outs = torch.cat(outs,dim=1)
		outs = outs.view(-1,outs.size(1))
		outs = self.linear(outs)
		return outs



