import argparse
import dgl
import numpy
import dgl.function as fn
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# from dgl.nn.pytorch import SAGEConv
from dgl.utils import expand_as_pair
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import tqdm
import sys
# sys.path.insert(0,'..')
from utils import Logger
from memory_usage import see_memory_usage, nvidia_smi_usage

from cpu_mem_usage import get_memory

#! the cahce and reuse layer 
class SAGEConvCacheReuse(nn.Module):
	def __init__(self,
				 in_feats,
				 out_feats,
				 aggregator_type,
				 bias=False
				 ):

		super(SAGEConvCacheReuse, self).__init__()

		self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
		self._out_feats = out_feats
		self._aggre_type = aggregator_type
		# aggregator type: mean/pool/lstm/gcn
		if aggregator_type == 'pool':
			self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
		if aggregator_type == 'lstm':
			self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
		if aggregator_type != 'gcn':
			self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
		# self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
		self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
		self.reset_parameters()

		#//self.embeddings = torch.empty((1,12345))  #TODO adjust the length 

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		# gain = nn.init.calculate_gain('relu')
		# nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
		# nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
		gain = nn.init.calculate_gain('relu')
		if self._aggre_type == 'pool':
			nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
		if self._aggre_type == 'lstm':
			self.lstm.reset_parameters()
		if self._aggre_type != 'gcn':
			nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
		nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
	
	def _lstm_reducer(self, nodes):
		"""LSTM reducer
		NOTE(zihao): lstm reducer with default schedule (degree bucketing)
		is slow, we could accelerate this with degree padding in the future.
		"""
		# print(nodes)
		m = nodes.mailbox['m'] # (B, L, D)
		# print('m.shape '+str(m.shape))
		# see_memory_usage("----------------------------------------1")
		batch_size = m.shape[0]
		# see_memory_usage("----------------------------------------2")
		h = (m.new_zeros((1, batch_size, self._in_src_feats)),
			 m.new_zeros((1, batch_size, self._in_src_feats)))
		# print(' h.shape '+ str(h[0].shape)+', '+ str(h[1].shape))
		# see_memory_usage("----------------------------------------3")
		
		_, (rst, _) = self.lstm(m, h)
		# see_memory_usage("----------------------------------------4")
		# print('rst.shape ',rst.shape)
		return {'neigh': rst.squeeze(0)}



	def forward(self, graph, feat, prev_layer_repeat, step, flag, reuse_embedding):
		r"""Compute GraphSAGE layer.
		Parameters
		----------
		graph : DGLGraph
			The graph.
		feat : torch.Tensor or pair of torch.Tensor
			If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
			:math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
			If a pair of torch.Tensor is given, the pair must contain two tensors of shape
			:math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
		Returns
		-------
		torch.Tensor
			The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
			is size of output feature.
		"""
		#print("block_size", graph.srcdata[dgl.NID].size(), feat.size())
		graph = graph.local_var()

		if isinstance(feat, tuple):
			feat_src, feat_dst = feat
		else:
			feat_src = feat_dst = feat
		if isinstance(feat, tuple):
			feat_src, feat_dst = feat
		else:
			feat_src = feat_dst = feat
			if graph.is_block:
				feat_dst = feat_src[:graph.number_of_dst_nodes()]
		
		msg_fn = fn.copy_src('h', 'm')
		h_self = feat_dst

		#print('node_num', graph.number_of_dst_nodes(), graph.number_of_src_nodes(), feat.size())

		if self._aggre_type == 'mean':
			graph.srcdata['h'] =  feat_src
			graph.update_all(msg_fn, fn.mean('m', 'neigh'))
			h_neigh = graph.dstdata['neigh']
			h_neigh = self.fc_neigh(h_neigh)
		# graph.srcdata['h'] = feat_src
		# graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
		# h_neigh = graph.dstdata['neigh']
		elif self._aggre_type == 'pool':
			graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
			graph.update_all(msg_fn, fn.max('m', 'neigh'))
			h_neigh = self.fc_neigh(graph.dstdata['neigh'])
		elif self._aggre_type == 'lstm':
			graph.srcdata['h'] = feat_src
			# see_memory_usage("----------------------------------------before graph.update_all(msg_fn, self._lstm_reducer)")
			graph.update_all(msg_fn, self._lstm_reducer)
			# see_memory_usage("----------------------------------------after graph.update_all")
			
			h_neigh = self.fc_neigh(graph.dstdata['neigh'])
			# see_memory_usage("----------------------------------------after h_neigh = self.fc_neigh")
		rst = self.fc_self(h_self) + h_neigh

		
		# if flag == 0:
		# 	src_nids = graph.srcdata[dgl.NID]
		# 	src_indices = np.arange(len(src_nids))   #toloist()
		# 	reuse_indices = prev_layer_repeat[step][0]
		# 	unprune_indices = np.delete(src_indices, reuse_indices)
		# 	reuse_nodes = src_nids[reuse_indices]  #reuse for src
		# 	print("reuse_nodes", reuse_nodes)
		# 	#full_feat=torch.Tensor(len(src_nids), feat.size(1)).to(graph.device) #
		# 	#print("feat_type", full_feat.dtype, feat.dtype,reuse_embedding.dtype)
			
		# 	if reuse_embedding.size(0)>1: 
		# 		full_feat[reuse_indices] = reuse_embedding
		# 	feat = full_feat

		#*flag = 0 for reuse; 1 for cache
		
		#! the last layer not only for cache for the next batch but also reuse for next layer
		#! add cache embedding to src feature
		if flag==0:
			cache_embedding=torch.tensor([], dtype=torch.float32, device=graph.device) #.to(graph.device)
			#if flag == 1 :  
			reuse_indices = prev_layer_repeat[step][1] 
			if len(reuse_indices)>0:  # some nodes need reuse
				full_dst_len = rst.size(0)+len(reuse_indices)
				full_dst_indices = np.arange(full_dst_len) 
				#print("len_dst", len(full_dst_indices), len(reuse_indices), rst.size())
				unprune_indices = np.delete(full_dst_indices, reuse_indices)
				tic = time.time()
				#full_dst_feat=temp_emb[full_dst_indices]
				full_dst_feat=torch.full((full_dst_len, rst.size(1)), 0, dtype=torch.float32, device=graph.device)
				#full_dst_feat=torch.tensor(full_dst_len, rst.size(1), device=graph.device) #.to(graph.device) #
				tic1 = time.time()
				print(f"tensor_cre: {tic1-tic}")
				full_dst_feat[unprune_indices] = rst  #avoid the none cannot assigment
				#if reuse_embedding.size(0)>1: 
				#print("reuse_size", reuse_embedding.size())
				full_dst_feat[reuse_indices] = reuse_embedding
				tic2 = time.time()
				print(f"concat_tensor: {tic2-tic1}")
				rst = full_dst_feat
				#print("rst.size", rst.size())

		#if flag == 1:
			try:
				tic3 = time.time()
				#//dst_nids = graph.dstdata[dgl.NID] the dst is incomplete
				cache_indices = prev_layer_repeat[step+1][0]  #* cache indice in current batch 
				##/cache_nodes = dst_nids[cache_indices]  #cache for the dst
				#print("cache_nodes", len(cache_indices), rst.size())
				#if len(cache_indices)>0:
				cache_embedding = rst[cache_indices]
				cache_embedding = cache_embedding.detach() 
				print(f"slice_tensor: {time.time()-tic3}") 
			except:
				pass
			#print("cache_size", cache_embedding.size())
			#print("featshape",self.fc_self(h_self).size(),h_neigh.size(),rst.size())
			# see_memory_usage("----------------------------------------after rst")

		if flag==0:  #! for training
			return rst, cache_embedding
		else:
			return rst


class SAGEConv(nn.Module):
	def __init__(self,
				 in_feats,
				 out_feats,
				 aggregator_type,
				 bias=False
				 ):

		super(SAGEConv, self).__init__()

		self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
		self._out_feats = out_feats
		self._aggre_type = aggregator_type
		# aggregator type: mean/pool/lstm/gcn
		if aggregator_type == 'pool':
			self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
		if aggregator_type == 'lstm':
			self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
		if aggregator_type != 'gcn':
			self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
		# self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
		self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
		self.reset_parameters()

		#//self.embeddings = torch.empty((1,12345))  #TODO adjust the length 

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		# gain = nn.init.calculate_gain('relu')
		# nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
		# nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
		gain = nn.init.calculate_gain('relu')
		if self._aggre_type == 'pool':
			nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
		if self._aggre_type == 'lstm':
			self.lstm.reset_parameters()
		if self._aggre_type != 'gcn':
			nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
		nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
	
	def _lstm_reducer(self, nodes):
		"""LSTM reducer
		NOTE(zihao): lstm reducer with default schedule (degree bucketing)
		is slow, we could accelerate this with degree padding in the future.
		"""
		# print(nodes)
		m = nodes.mailbox['m'] # (B, L, D)
		# print('m.shape '+str(m.shape))
		# see_memory_usage("----------------------------------------1")
		batch_size = m.shape[0]
		# see_memory_usage("----------------------------------------2")
		h = (m.new_zeros((1, batch_size, self._in_src_feats)),
			 m.new_zeros((1, batch_size, self._in_src_feats)))
		# print(' h.shape '+ str(h[0].shape)+', '+ str(h[1].shape))
		# see_memory_usage("----------------------------------------3")
		
		_, (rst, _) = self.lstm(m, h)
		# see_memory_usage("----------------------------------------4")
		# print('rst.shape ',rst.shape)
		return {'neigh': rst.squeeze(0)}



	def forward(self, graph, feat):
		r"""Compute GraphSAGE layer.
		Parameters
		----------
		graph : DGLGraph
			The graph.
		feat : torch.Tensor or pair of torch.Tensor
			If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
			:math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
			If a pair of torch.Tensor is given, the pair must contain two tensors of shape
			:math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
		Returns
		-------
		torch.Tensor
			The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
			is size of output feature.
		"""
		graph = graph.local_var()

		if isinstance(feat, tuple):
			feat_src, feat_dst = feat
		else:
			feat_src = feat_dst = feat
		if isinstance(feat, tuple):
			feat_src, feat_dst = feat
		else:
			feat_src = feat_dst = feat
			if graph.is_block:
				feat_dst = feat_src[:graph.number_of_dst_nodes()]
		
		msg_fn = fn.copy_src('h', 'm')
		h_self = feat_dst

		#print('node_num', graph.number_of_dst_nodes(), graph.number_of_src_nodes(), feat.size())

		if self._aggre_type == 'mean':
			graph.srcdata['h'] =  feat_src
			graph.update_all(msg_fn, fn.mean('m', 'neigh'))
			h_neigh = graph.dstdata['neigh']
			h_neigh = self.fc_neigh(h_neigh)
		# graph.srcdata['h'] = feat_src
		# graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
		# h_neigh = graph.dstdata['neigh']
		elif self._aggre_type == 'pool':
			graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
			graph.update_all(msg_fn, fn.max('m', 'neigh'))
			h_neigh = self.fc_neigh(graph.dstdata['neigh'])
		elif self._aggre_type == 'lstm':
			graph.srcdata['h'] = feat_src
			# see_memory_usage("----------------------------------------before graph.update_all(msg_fn, self._lstm_reducer)")
			graph.update_all(msg_fn, self._lstm_reducer)
			# see_memory_usage("----------------------------------------after graph.update_all")
			
			h_neigh = self.fc_neigh(graph.dstdata['neigh'])
			# see_memory_usage("----------------------------------------after h_neigh = self.fc_neigh")

		rst = self.fc_self(h_self) + h_neigh
		#print("featshape",self.fc_self(h_self).size(),h_neigh.size(),rst.size())
		# see_memory_usage("----------------------------------------after rst")
		return rst



class GraphSAGE(nn.Module):
	def __init__(self,
				 in_feats,
				 hidden_feats,
				 out_feats,
				 aggre,
				 num_layers,
				 activation,
				 dropout):
		super(GraphSAGE, self).__init__()
		self.n_hidden = hidden_feats
		self.n_classes = out_feats
		self.activation = activation

		self.layers = nn.ModuleList()
		# self.bns = nn.ModuleList()
		#! the second last layers  should use the SAGEConvCacheReuse
		#* the last one for Cache the GNN embedding and the output layer for Reuse
		if num_layers==1:  #should cache the features
			#self.layers.append(SAGEConvCacheReuse(in_feats, out_feats aggre, bias=False))
			self.layers.append(SAGEConvCacheReuse(in_feats, out_feats, aggre, bias=False)) 
		elif num_layers==2: #
			self.layers.append(SAGEConvCacheReuse(in_feats, hidden_feats, aggre, bias=False))
			self.layers.append(SAGEConv(hidden_feats, out_feats, aggre, bias=False))
		else:
			# input layer
			self.layers.append(SAGEConv(in_feats, hidden_feats, aggre, bias=False))
			#self.layers.append(SAGEConvCacheReuse(hidden_feats, hidden_feats, aggre, bias=False))
			# self.bns.append(nn.BatchNorm1d(hidden_feats))
			# hidden layers
			for _ in range(1, num_layers - 2):
				self.layers.append(SAGEConv(hidden_feats, hidden_feats, aggre, bias=False))
				# self.bns.append(nn.BatchNorm1d(hidden_feats))
			# output layer
			self.layers.append(SAGEConvCacheReuse(hidden_feats, hidden_feats, aggre, bias=False))
			self.layers.append(SAGEConv(hidden_feats, out_feats, aggre, bias=False))
		self.dropout = nn.Dropout(p=dropout)

	def reset_parameters(self):
		for layer in self.layers:
			layer.reset_parameters()
		# for bn in self.bns:
		# 	bn.reset_parameters()

	def forward(self, blocks_and_repeat, input_reuse):
		blocks = blocks_and_repeat[0]
		prev_layer_repeat = blocks_and_repeat[1]
		step = blocks_and_repeat[2]
		x=input_reuse[0]
		reuse_embedding = input_reuse[1]
		for i, (layer, block) in enumerate(zip(self.layers[:-1], blocks[:-1])):
			#tic = time.time()
		# for i, layer in enumerate(self.layers[:-1]):
			l = len(blocks)
			#print('layer', i)
			# see_memory_usage("----------------------------------------before model layer "+str(i))
			# print(x.shape)
			if (i == l-2):  #(i == l-1) or
				#! flag 0 for trianing; 1 for inference
				x, cache_embedding = layer(block, x, prev_layer_repeat, step, 0, reuse_embedding)  #* the last layer
			else:
				x = layer(block, x)
			#print(f"layer{i}: {time.time()-tic}")
			# see_memory_usage("----------------------------------------after model layer "+str(i)+ ' x = layer(block, x)')
			# print(x.shape)
			# if i==0:
			# 	print("first layer input nodes number: "+str(len(block.srcdata[dgl.NID])))
			# 	print("first layer output nodes number: "+str(len(block.dstdata[dgl.NID])))
			# else:
			# 	print("input nodes number: "+str(len(block.srcdata[dgl.NID])))
			# 	print("output nodes number: "+str(len(block.dstdata[dgl.NID])))
			# print("edges number: "+str(len(block.edges()[1])))
			# print("input nodes : "+str((blocks[-1].srcdata[dgl.NID])))
			# print("output nodes : "+str((blocks[-1].dstdata[dgl.NID])))
			# print("edges number: "+str((blocks[-1].edges())))
			# print("dgl.NID: "+str(dgl.NID))
			# print("dgl.EID: "+str((dgl.EID)))

			x = self.activation(x)
			# see_memory_usage("----------------------------------------after model layer "+str(i)+ " x = self.activation(x)")
			# print(x.shape)

			x = self.dropout(x)
			# see_memory_usage("----------------------------------------after model layer "+str(i)+' x = self.dropout(x)')
			# print(x.shape)
		#tic1 = time.time()
		x= self.layers[-1](blocks[-1], x)  #* the output layer
		#print(f"layer{i+1}: {time.time()-tic1}")
		# see_memory_usage("----------------------------------------end of model layers  after x = self.layers[-1](blocks[-1], x)")
		# print(x.shape)
		# print("input nodes number: "+str(len(blocks[-1].srcdata[dgl.NID])))
		# print("output nodes number: "+str(len(blocks[-1].dstdata[dgl.NID])))
		# print("edges number: "+str(len(blocks[-1].edges()[1])))
		# print("input nodes : "+str((blocks[-1].srcdata[dgl.NID])))
		# print("output nodes : "+str((blocks[-1].dstdata[dgl.NID])))
		# print("edges number: "+str((blocks[-1].edges())))
		return x.log_softmax(dim=-1), cache_embedding

	def inference(self, g, x, args, device):
		"""
		Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
		g : the entire graph.
		x : the input of entire node set.

		The inference code is written in a fashion that it could handle any number of nodes and
		layers.
		"""
		# During inference with sampling, multi-layer blocks are very inefficient because
		# lots of computations in the first few layers are repeated.
		# Therefore, we compute the representation of all nodes layer by layer.  The nodes
		# on each layer are of course splitted in batches.
		# TODO: can we standardize this?
		device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
		for l, layer in enumerate(self.layers):
			print("cur_layer:", l)
			y = torch.zeros(g.num_nodes(), self.n_hidden if l!=len(self.layers) - 1 else self.n_classes)

			sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
			dataloader = dgl.dataloading.NodeDataLoader(
				g,
				torch.arange(g.num_nodes(),dtype=torch.long).to(g.device),
				sampler,
				device=device,
				# batch_size=24,
				batch_size=args.batch_size,
				shuffle=True,
				drop_last=False,
				num_workers=args.num_workers)

			for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):

				block = blocks[0]
				block = block.int().to(device)
				h = x[input_nodes].to(device)
			
				if l==len(self.layers)-2:
					h = layer(block, h, None, 0, 1, None)  #* same as the forward
				else:
					h = layer(block, h)


				# y[output_nodes] = h
				y[output_nodes] = h.cpu()

			x = y
		return y
