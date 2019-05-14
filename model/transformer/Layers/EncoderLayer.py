# __author__ = 'zhuqing'
import torch.nn as nn
import torch
from model.transformer.SubLayers.Multi_head_att import MultiHeadAttention
from model.transformer.SubLayers.PositionwiseFeedForward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
	def __init__(self,dim_model,dim_inner,nums_head,d_k,d_v,dropout=0.1):
		super(EncoderLayer,self).__init__()
		self.self_attn = MultiHeadAttention(
			nums_head=nums_head,dim_model=dim_model,d_k=d_k,d_v=d_v,dropout=dropout)
		self.pos_ffn = PositionwiseFeedForward(
			d_in=dim_model,d_hidden = dim_inner,dropout=dropout)

	def forward(self,encoder_input,non_pad_mask=None,self_attn_mask=None):
		encoder_output ,encoder_self_attn = self.self_attn(
			encoder_input,encoder_input,encoder_input,mask=self_attn_mask)
		encoder_output *= non_pad_mask

		encoder_output = self_attn.pos_ffn(encoder_output)
		encoder_output *= non_pad_mask

		return encoder_output,encoder_self_attn












