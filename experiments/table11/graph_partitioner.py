import numpy 
import dgl
from numpy.core.numeric import Infinity
import multiprocessing as mp
import torch
import time
from statistics import mean
from my_utils import *
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import cupy as cp
import sys
from collections import Counter
from math import ceil
from cpu_mem_usage import get_memory
import gc

# sys.path.insert(0,'..')
# from draw_graph import draw_dataloader_blocks_pyvis_total, gen_pyvis_graph_local

class Graph_Partitioner:  # ----------------------*** split the output layer block ***---------------------
	def __init__(self, layer_block, args):
		# self.balanced_init_ratio=args.balanced_init_ratio
		self.dataset=args.dataset
		self.layer_block=layer_block # local graph with global nodes indices
		self.local=False
		self.output_nids=layer_block.dstdata['_ID'] # tensor type
		self.local_output_nids=[]
		self.local_src_nids=[]
		self.src_nids_list= layer_block.srcdata['_ID'].tolist()
		self.full_src_len=len(layer_block.srcdata['_ID'])
		self.global_batched_seeds_list=[]
		self.local_batched_seeds_list=[]
		self.weights_list=[]
		# self.alpha=args.alpha 
		# self.walkterm=args.walkterm
		self.num_batch=args.num_batch
		self.selection_method=args.selection_method
		self.batch_size=0
		self.ideal_partition_size=0

		# self.bit_dict={}
		self.side=0
		self.partition_nodes_list=[]
		self.partition_len_list=[]

		self.time_dict={}
		self.red_before=[]
		self.red_after=[]
		self.args=args
		self.re_part_block=[]
		
		
	def gen_batched_seeds_list(self):
		'''
		Parameters
		----------
		OUTPUT_NID: final layer output nodes id (tensor)
		selection_method: the graph partition method

		Returns
		-------
		'''
	
		full_len = len(self.local_output_nids)  # get the total number of output nodes
		self.batch_size=get_mini_batch_size(full_len,self.num_batch)
		
		indices=[]
		# if 'range' in self.selection_method: #'range_init_graph_partition' 
		# 	t=time.time()
		# 	indices = [i for i in range(full_len)]
		# 	batches_nid_list, weights_list=gen_batch_output_list(self.local_output_nids,indices,self.batch_size)
		# 	print('range_init for graph_partition spend: ', time.time()-t)
		# elif 'random' in self.selection_method : #'random_init_graph_partition' 
		# 	t=time.time()
		# 	indices = random_shuffle(full_len)
		# 	batches_nid_list, weights_list=gen_batch_output_list(self.local_output_nids,indices,self.batch_size)
		# 	print('random_init for graph_partition spend: ', time.time()-t)
		# elif 'balance' in self.selection_method: #'balanced_init_graph_partition' 
		# 	t=time.time()
		# 	batches_nid_list, weights_list=self.balanced_init()
		# 	print('balanced_init for graph_partition spend: ', time.time()-t)
		if 'metis' in self.selection_method:
			o_graph = self.args.o_graph
			partition = dgl.metis_partition(g=o_graph,k=self.args.num_batch)
			# print('---------------pure dgl.metis_partition spent ', time.time()-t8 )
			res=[]
			for pid in partition:
				nids = partition[pid].ndata[dgl.NID].tolist()
				res.append(sorted(nids))
				print(len(nids))
			if set(sum(res,[]))!=set(self.local_output_nids):
				print('--------pure    check:     the difference of graph partition res and self.local_output_nids')
			batches_nid_list = res
			weights_list = []
			output_num = len(self.local_output_nids)
			for pid_list in res:
				pid_len = len(pid_list)
				weights_list.append(pid_len/output_num)
		else:
			print('\t\t\t error in seletion method !!!')
			
		
		self.local_batched_seeds_list=batches_nid_list
		
		self.weights_list=weights_list

		print('The batched output nid list before graph partition')
		# print_len_of_batched_seeds_list(batches_nid_list)

		return 



	def remove_non_output_nodes(self):    #from src remove the node not in dst
		import copy
		local_src=copy.deepcopy(self.local_src_nids)
		#print('local_src', self.local_src_nids)
		#print('local_output', self.local_output_nids)
		mask_array = np.full(len(local_src),True, dtype=np.bool)
		mask_array[self.local_output_nids] = False  # the indice of output node is false
		from itertools import compress
		to_remove=list(compress(local_src, mask_array)) # time complexity O(n)  the from the local_src remove local_output
		print('len', len(local_src),len(self.local_output_nids),len(to_remove))
		return to_remove

	def get_src(self, seeds):
		in_ids=list(self.layer_block.in_edges(seeds))[0].tolist()
		src= list(set(in_ids+seeds))
		return src



	def simple_gen_K_batches_seeds_list(self):  #! the batch level partition here 
		
		if self.selection_method == "random" or self.selection_method == "range" or self.selection_method=="metis":
			self.gen_batched_seeds_list()
		elif self.selection_method == "REG" :
			print('REG start----................................')
			ts = time.time()
			get_memory('---------================-----------------=============---------REG before start \n')
			u, v =self.layer_block.edges()[0], self.layer_block.edges()[1] # local edges
			print("v_edges",self.layer_block.edges()[1]) #local
			#print("dst_nodes",self.layer_block.dst_nodes()) #global
			#print("nodes",self.layer_block.nodes())
			#np.save('layer_block_edges.npy', self.layer_block.edges())
			#print('u,v',len(u),type(u))  #
			g = dgl.graph((u,v))
			#np.save('u.npy', u)
			#np.save('v.npy', v)
			# graph_in = Counter(g.in_degrees().tolist())
			# print('number of edges of full batch : ',len(u) )
			# get_memory('---------================-----------------=============---------REG start \n')
			A = g.adjacency_matrix() 
			#print('A',A.size())
			#print('A._indices',A._indices().size())
			# get_memory('---------================-----------------=============---------REG A \n')
			AT= torch.transpose(A, 0, 1)
			# get_memory('---------================-----------------=============---------REG AT \n')
			m_at = AT._indices().tolist()  #? _indices is the id of src and dst
			#print("AT",AT.size())
			#print("m_at", len(m_at))
			# get_memory('---------================-----------------=============---------REG indices AT \n')
			m_a  = A._indices().tolist()
			length = len(m_a[0])
			length2 = len(m_a[1])
			#print("m_a", len(m_a[0]), len(m_a[1]))
			# get_memory('---------================-----------------=============---------REG indices A \n')
			g_at = dgl.graph((m_at[0], m_at[1]))   #! is the same as AT  (why do this?)
			g_at.edata['w'] = torch.ones(length).requires_grad_()
			# get_memory('---------================-----------------=============---------REG weight AT \n')
			g_a = dgl.graph((m_a[0], m_a[1]))   #! is the same as A  
			g_a.edata['w'] = torch.ones(length).requires_grad_()
			# get_memory('---------================-----------------=============---------REG weight A \n')
			auxiliary_graph = dgl.adj_product_graph(g_at, g_a, 'w')   #* this adj record the shared neighbors of two nodes 
			get_memory('---------================-----------------=============---------REG auxiliary graph done \n')
			to_remove = self.remove_non_output_nodes()   
			if len(to_remove)>0:
				auxiliary_graph.remove_nodes(torch.tensor(to_remove))    # keep the neighbor relation of output nodes, no need for other nodes
																		# because partition among output nodes, so no repeated
			#! only need to know the retationship between output nodes (train nodes), so remove other nodes from the graph
			auxiliary_graph_no_diag = dgl.remove_self_loop(auxiliary_graph)

			#!adj_matrix sparse
			# print("before auxiliary_graph", auxiliary_graph_no_diag.adjacency_matrix().size(),auxiliary_graph_no_diag.adjacency_matrix()) #
			# edges_weight = auxiliary_graph_no_diag.edata['w']
			# mean_degree = torch.mean(edges_weight)  #* 
			# #print("before weight", auxiliary_graph_no_diag.edata['w'])
			# #print("mean", mean_degree)
			# edges_weight[edges_weight<mean_degree] = 0
			# auxiliary_graph_no_diag.edata['w'] = edges_weight
			# removed_edges = torch.nonzero(edges_weight==0).squeeze()   #fine the 0 edges
			# print('removed edges', removed_edges, removed_edges.size())
			# auxiliary_graph_no_diag.remove_edges(removed_edges)
			# print("after auxiliary_graph", auxiliary_graph_no_diag.adjacency_matrix().size(),auxiliary_graph_no_diag.adjacency_matrix()) #

			# res = Counter(auxiliary_graph_no_diag.edata['w'].tolist())
			tp1 = time.time()
			#print("m_a", len(m_a[0]), len(m_a[1]))
			print("num_part", self.args.num_batch)
			get_memory('before Metis partition \n')
			partition = dgl.metis_partition(g=auxiliary_graph_no_diag,k=self.args.num_batch)
			get_memory('after Metis partition \n')
			tp2 = time.time()
			res=[]
			out_len=0
			print("dstlen", len(self.layer_block.dstdata[dgl.NID]))
			for pid in partition:
				nids = partition[pid].ndata[dgl.NID].tolist()  #local id
				#print("partition", nids)
				res.append(sorted(nids))
				# print(len(nids))
				out_len+=len(nids)
				print("outnodes",len(nids),out_len)
			#print("local_batched_seeds_list",len(res))
			print('REG metis partition end ----................................')
			# print('the time spent: ', tp2-ts)
			print('REG construction  time spent: ', tp1-ts)
			print('pure dgl.metis_partition the time spent: ', tp2-tp1)
			self.local_batched_seeds_list=res
			#print('local_batched_seeds_list', len(res), type(res))
			# del g,A,AT,m_a,m_at,g_a,g_at,auxiliary_graph,auxiliary_graph_no_diag,partition
			# gc.collect()
		return

		# if self.selection_method == "REG" :
			
		# 	t1=time.time()
		# 	u, v =self.layer_block.edges()[0], self.layer_block.edges()[1] # local edges
		# 	g = dgl.graph((u,v))
		# 	print('g = dgl.graph((u,v))  spent ', time.time()-t1 )

		# 	print('the counter of in-degree current block !!!!!!!!!!!!!!_______________!!!!!!!!!!')
		# 	graph_in = Counter(g.in_degrees().tolist())
		# 	print(graph_in)
		# 	print()
		# 	t13 = time.time()
		# 	A = g.adjacency_matrix()
		# 	print('A = g.adjacency_matrix() spent ', time.time()-t13 )
			
		# 	AT= torch.transpose(A, 0, 1)
			
		# 	m_at = AT._indices().tolist()
		# 	m_a  = A._indices().tolist()
		# 	length = len(m_a[0])
			
		# 	g_at = dgl.graph((m_at[0], m_at[1]))
		# 	g_at.edata['w'] = torch.ones(length).requires_grad_()
		# 	g_a = dgl.graph((m_a[0], m_a[1]))
		# 	g_a.edata['w'] = torch.ones(length).requires_grad_()
		# 	auxiliary_graph = dgl.adj_product_graph(g_at, g_a, 'w')
		# 	print('auxiliary_graph')
		# 	print(auxiliary_graph)
		
		# 	t2 = time.time()
		# 	remove = self.remove_non_output_nodes() # use mask to replace the for loop to speed up
		# 	print('get remove nodes spent ', time.time()-t2 )
		# 	print('remove nodes length ', len(remove))
		# 	print()
			
		# 	if len(remove)>0:
		# 		t3 = time.time()
		# 		auxiliary_graph.remove_nodes(torch.tensor(remove))
		# 		print('auxiliary_graph.remove_nodes spent ', time.time()-t3 )
		# 		print('after remove non output nodes the auxiliary_graph')
		# 		print(auxiliary_graph)
			
		# 	t7 = time.time()
		# 	auxiliary_graph_no_diag = dgl.remove_self_loop(auxiliary_graph)
		# 	print('auxiliary_graph_no_diag generation spent ', time.time()-t7 )
		
		# 	print()
		# 	print('the counter of shared neighbor distribution')
		# 	res = Counter(auxiliary_graph_no_diag.edata['w'].tolist())
		# 	print(res)
			
		# 	ll=len(auxiliary_graph_no_diag.edata['w'])
		# 	print(len(auxiliary_graph_no_diag.edata['w']))
			

		# 	t8 = time.time()
		# 	partition = dgl.metis_partition(g=auxiliary_graph_no_diag,k=self.args.num_batch)
		# 	print('auxiliary_graph_no_diag dgl.metis_partition spent ', time.time()-t8 )
		# 	res=[]
		# 	for pid in partition:
		# 		nids = partition[pid].ndata[dgl.NID].tolist()
		# 		res.append(sorted(nids))
		# 		print(len(nids))
		# 	if set(sum(res,[]))!=set(self.local_output_nids):
		# 		print('the difference of graph partition res and self.local_output_nids')
				
			
		# 	self.local_batched_seeds_list=res
			
		# 	return 




	def get_src_len(self,seeds):
		in_ids=list(self.layer_block.in_edges(seeds))[0].tolist()
		src_len= len(list(set(in_ids+seeds)))
		return src_len



	def get_partition_src_len_list(self):
		partition_src_len_list=[]
		for seeds_nids in self.local_batched_seeds_list:
			partition_src_len_list.append(self.get_src_len(seeds_nids))
		
		self.partition_src_len_list=partition_src_len_list
		return partition_src_len_list


	def graph_partition(self):
		
		# full_batch_subgraph=self.layer_block #heterogeneous graph (block)
		
		# print('----------------------------  graph partition start---------------------')
		
		self.ideal_partition_size = (self.full_src_len/self.num_batch)
		
		t2 = time.time()
		
		self.simple_gen_K_batches_seeds_list()
		# print('total k batches seeds list generation spend ', time.time()-t2 )

		weight_list = get_weight_list(self.local_batched_seeds_list)
		src_len_list = self.get_partition_src_len_list()

		# print('after graph partition')
		
		self.weights_list = weight_list
		self.partition_len_list = src_len_list

		return self.local_batched_seeds_list, weight_list, src_len_list
	


	def global_to_local(self):
		
		sub_in_nids = self.src_nids_list
		# print('src global')
		# print(sub_in_nids)#----------------
		# global_nid_2_local = {sub_in_nids[i]: i for i in range(0, len(sub_in_nids))}
		global_nid_2_local = dict(zip(sub_in_nids,range(len(sub_in_nids))))
		self.local_output_nids = list(map(global_nid_2_local.get, self.output_nids.tolist()))
		# print('dst local')
		# print(self.local_output_nids)#----------------
		self.local_src_nids = list(map(global_nid_2_local.get, self.src_nids_list))
		
		self.local=True
		return 	


	def local_to_global(self):
		sub_in_nids = self.src_nids_list
		# local_nid_2_global = { i: sub_in_nids[i] for i in range(0, len(sub_in_nids))}
		local_nid_2_global = dict(zip(range(len(sub_in_nids)), sub_in_nids))
		global_batched_seeds_list=[]
		for local_in_nids in self.local_batched_seeds_list:
			global_in_nids = list(map(local_nid_2_global.get, local_in_nids))
			global_batched_seeds_list.append(global_in_nids)

		self.global_batched_seeds_list=global_batched_seeds_list
		# print('-----------------------------------------------global batched output nodes id----------------------------')
		# for inp in self.global_batched_seeds_list:
		# 	print(len(sorted(inp)))
		# print(self.global_batched_seeds_list)
		self.local=False
		return 


	def init_graph_partition(self):
		ts = time.time()
		
		self.global_to_local() # global to local            self.local_batched_seeds_list
		# print('global_2_local spend time (sec)', (time.time()-ts))
		# print()
		
		t2=time.time()
		# Then, the graph_parition is run in block to graph local nids,it has no relationship with raw graph
		self.graph_partition()
		# print('graph partition algorithm spend time', time.time()-t2)
		# print()
		# after that, we transfer the nids of batched output nodes from local to global.
		self.local_to_global() # local to global         self.global_batched_seeds_list
		t_total=time.time()-ts

		#print("return", self.global_batched_seeds_list[0],self.weights_list[0],self.partition_len_list[0])

		return self.global_batched_seeds_list, self.weights_list, t_total, self.partition_len_list