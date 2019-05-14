# __author__ = 'zhuqing'
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.transformer.Attention.ScaleProDotAtt import ScaleProductDotAttention
class MultiHeadAttention(nn.Module):
	'''
	参数描述
	'''
	def __init__(self,nums_head,dim_model,d_k,d_v,dropout):
		super().__init__()
		self.nums_head = nums_head
		self.d_k = d_k
		self.d_v = d_v

		self.w_q = nn.Linear(dim_model,nums_head * d_k)
		self.w_k = nn.Linear(dim_model,nums_head * d_k)
		self.w_v = nn.Linear(dim_model,nums_head * d_v)
		#initializing
		nn.init.normal_(self.w_q.weight,mean = 0,std=np.sqrt(2.0/(dim_model+d_k)))
		nn.init.normal_(self.w_k.weight,mean = 0,std=np.sqrt(2.0/(dim_model+d_k)))
		nn.init.normal_(self.w_v.weight,mean = 0,std=np.sqrt(2.0/(dim_model+d_v)))

		self.attention = ScaleProductDotAttention(temperature=np.power(d_k,0.5))
		self.layer_norm = nn.LayerNorm(dim_model)

		self.fc = nn.Linear(nums_head * d_v , dim_model)
		nn.init.xavier_normal_(self.fc.weight)
		self.dropout = nn.Dropout(dropout)

	def forward(self,q,k,v,mask=None):
		d_k ,d_v , nums_head = self.d_k,self.d_v,self.nums_head
		sz_b , len_q, _ = q.size()
		sz_b , len_k, _ = k.size()
		sz_b , len_v, _ = v.size()

		#残差
		residual = q
		#get q,k,v
		q = self.wq(q).view(sz_b,len_q,nums_head,d_k)
		k = self.wq(k).view(sz_b,len_k,nums_head,d_k)
		v = self.wq(v).view(sz_b,len_v,nums_head,d_v)

		q = q.permute(2,0,1,3).contiguous().view(-1,len_q,d_k)
		k = k.permute(2,0,1,3).contiguous().view(-1,len_k,d_k)
		v = v.permute(2,0,1,3).contiguous().view(-1,len_v,d_v)

		mask = mask.repeat(nums_head,1,1)
		output,attn = self.attention(q,k,v,mask = mask)

		output = output.view(nums_head,sz_b,len_q,d_v)
		output = output.permute(1,2,0,3).contiguous().view(sz_b,len_q,-1)

		output = self.fc(output)
		output = self.dropout(output)
		output = self.layer_norm(output+residual)

		return output,attn

