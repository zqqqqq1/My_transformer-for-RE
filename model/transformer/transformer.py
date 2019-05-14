import torch
import torch.nn as nn
import numpy as np
from model.transformer.Layers.EncoderLayer import EncoderLayer
from model.transformer.Layers.DecoderLayer import DecoderLayer
from model.transformer import utils
from model.transformer.utils import get_absolute_position_table
from model.transformer.utils import get_non_pad_mask
from model.transformer.utils import get_attn_key_pad_mask
from model.transformer.utils import get_subsequent_mask
from model.transformer.CNN import CNN
class Encoder(nn.Module):
	def __init__(
		self,
		n_src_vocab,len_max_seq,d_word_vec,
		n_layers,n_head,d_k,d_v,d_model,
		d_inner,dropout=0.1):
		super().__init__()
		n_position = len_max_seq +1
		self.src_word_emb = nn.Embedding(
			n_src_vocab,d_word_vec,padding_idx = utils.PAD)

		self.absolute_position_enc=nn.Embedding.from_pretrained(
			get_absolute_position_table(n_position,d_word_vec,padding_idx=0),
			freeze=True)

		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model,d_inner,n_head,d_k,d_v,dropout)
			for _ in range(n_layers)])
	def forward(self,src_seq,src_pos,return_attn=False):
		enc_self_attn_list = []

		self_attn_mask = get_attn_key_pad_mask(seq_k=src_seq,seq_q=src_seq)
		non_pad_mask = get_non_pad_mask(src_seq)

		enc_output = self.src_word_emb(src_seq)+self.absolute_position_enc(src_pos)

		for enc_layer in self.layer_stack:
			enc_output,enc_self_attn = enc_layer(
				enc_output,non_pad_mask=non_pad_mask,
				self_attn_mask=self_attn_mask)
			if return_attn:
				enc_self_attn_list+=[enc_self_attn]
		if return_attn:
			return enc_output,enc_self_attn_list
		return enc_output
class Decoder(nn.Module):
	def __init__(
		self,n_tgt_vocab,len_max_seq,d_word_vec,
		n_layers,n_head,d_k,d_v,
		d_model,d_inner,dropout=0.1):
		super().__init__()
		n_position = len_max_seq+1
		self.tgt_word_emb = nn.Embedding(
			n_tgt_vocab,d_word_vec,padding_idx=utils.PAD)

		self.absolute_position_enc = nn.Embedding.from_pretrained(
			get_absolute_position_table(n_position,d_word_vec,padding_idx=0),
			freeze=True)

		self.layer_stack = nn.ModuleList([
			DecoderLayer(d_model,d_inner,n_head,d_k,d_v,dropout=dropout)
			for _ in range(n_layers)])

	def forward(self,tgt_seq,tgt_pos,src_seq,enc_output,return_attns=False):
		dec_self_attn_list,dec_enc_attn_list = [],[]

		non_pad_mask = get_non_pad_mask(tgt_seq)

		self_attn_mask_subseq = get_subsequent_mask(tgt_seq)
		self_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq,seq_q=tgt_seq)
		self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).get(0)

		dec_enc_attn_mask = get_attn_key_pad_mask(seq_k = src_seq,seq_q=tgt_seq)

		dec_output = self.tgt_word_emb(tgt_seq)+self.absolute_position_enc(tgt_pos)

		for dec_layer in self.layer_stack:
			dec_output,dec_self_attn,dec_enc_attn = dec_layer(
				dec_output,enc_output,
				non_pad_mask=non_pad_mask,
				self_attn_mask=self_attn_mask,
				dec_enc_attn_mask=dec_enc_attn_mask)
		if return_attn:
			return dec_output,dec_self_attn_list,dec_enc_attn_list
		return dec_output

class Transformer(nn.Module):
	def __init__(
		self,
		n_src_vocab=32,len_max_seq=100,
		d_word_vec=32,d_model=32,d_inner=256,
		n_layers=1,n_head=2,d_k=8,d_v=8,dropout=0.1,d_position=10,
		out_ch=16,num_class=50,
		k_size = [3,4,5],
		tgt_emb_prj_weight_sharing=True,
		emb_src_tgt_weight_sharing=True):
		super().__init__()
		self.encoder = Encoder(
			n_src_vocab=n_src_vocab,len_max_seq=len_max_seq,
			d_word_vec=d_word_vec,d_model=d_model,d_inner=d_inner,
			n_layers=n_layers,n_head=n_head,d_k=d_k,d_v=d_v,
			dropout=dropout)

		# self.decoder = Decoder(
		# 	n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
		# 	d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
		# 	n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
		# 	dropout=dropout)

		self.cnn = CNN(
			len_max_seq=len_max_seq,
			k_size=k_size,
			d_pos=d_position,
			in_ch = d_word_vec+d_position*2,
			out_ch = out_ch,
			num_class=num_class)
	def forward(self,src_seq,src_pos,pos1,pos2):
		enc_output,_ = self.encoder(src_seq,src_pos,pos1,pos2)
		reslut = self.cnn(enc_output,pos1,pos2)
		return result
