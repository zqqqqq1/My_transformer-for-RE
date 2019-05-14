# __author__ = 'zhuqing'
import torch.nn as nn
import torch
from model.transformer.SubLayers.Multi_head_att import MultiHeadAttention
from model.transformer.SubLayers.PositionwiseFeedForward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
	def __init__(self,dim_model,dim_inner,nums_head,d_k,d_v,dropout=0.1):
		super(EncoderLayer,self).__init__()
		self.encoder_attn = MultiHeadAttention(
			nums_head=nums_head,dim_model=dim_model,d_k=d_k,d_v=d_v,dropout=dropout)
		self.self_attn = MultiHeadAttention(
			nums_head=nums_head,dim_model=dim_model,d_k=d_k,d_v=d_v,dropout=dropout)
		self.pos_ffn = PositionwiseFeedForward(
			d_in=dim_model,d_hidden = dim_inner,dropout=dropout)

	def forward(self,decoder_input,encoder_output,non_pad_mask=None,self_attn_mask=None,dec_enc_attn_mask=None):
		decoder_output,decoder_self_attn = self.self_attn(
			decoder_input,decoder_input,decoder_input,mask = self_attn_mask)
		decoder_output *= non_pad_mask

		decoder_output,decoder_encoder_attn = self.encoder_attn(
			decoder_output,encoder_output,encoder_output,mask=dec_enc_attn_mask)
		decoder_output *= non_pad_mask

		decoder_output = self.pos_ffn(decoder_output)
		decoder_output *= non_pad_mask

		return decoder_output,decoder_self_attn,decoder_encoder_attn











