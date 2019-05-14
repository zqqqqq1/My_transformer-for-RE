#__author__ = 'zhuqing'
import torch
import torch.nn as nn
import numpy as np

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


def get_non_pad_mask(seq):
	assert seq.dim() == 2
	return seq.ne(PAD).type(torch.float).unsqueeze(-1)

def get_absolute_position_table(n_position,d_hid,padding_idx=None):
	def cal_angle(position,hid_idx):
		return position/np.power(10000,2*(hid_idx//2)/d_hid)
	def get_pos_ang_vec(position):
		return [cal_angle(position,hid_j) for hid_j in range(d_hid)]

	sinusoid_table = np.array([get_pos_ang_vec(pos_i) for pos_i in range(n_position)])

	sinusoid_table[:,0::2]=np.sin(sinusoid_table[:,0::2])
	sinusoid_table[:,1::2]=np.cos(sinusoid_table[:,1::2])

	if padding_idx is not None:
		sinusoid_table[padding_idx] = 0
	return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k,seq_q):
	len_q = seq_q.size(1)
	padding_mask = seq_k.eq(PAD)
	padding_mask = padding_mask.unsqueeze(1).expand(-1,len_q,-1)
	return padding_mask

def get_subsequent_mask(seq):
	sz_b ,len_s = seq.size()
	subsequent_mask = torch.triu(
		torch.ones((len_s,len_s),device=seq.device,dtype=torch.uint8),diagonal=1)
	subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b,-1,-1)
	return subsequent_mask
