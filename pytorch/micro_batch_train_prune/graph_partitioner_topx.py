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

import ctypes
from ctypes import *
from numpy.ctypeslib import ndpointer,as_array
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
			get_memory('---------================-----------------=============---------Topk  before start \n')
			#u, v =self.layer_block.edges()[0], self.layer_block.edges()[1] # local edges
			#print(type(self.layer_block.edges()[0]),type(self.layer_block))
			#np.save('layer_block_edges.npy', self.layer_block.edges())

			#! new partition method!!
			u, v =self.layer_block.edges()[0], self.layer_block.edges()[1] # local edges
			#print("nodeid", self.layer_block.srcdata[dgl.NID], self.layer_block.dstdata[dgl.NID])
			out_nodes = numpy.arange(len(self.layer_block.dstdata[dgl.NID]))
			len_out = len(out_nodes)
			blen_out = len(out_nodes)//self.num_batch     # thelength of outnodes in each batch 
			print('out_nodes',len(out_nodes), blen_out)
			print("edges",u,v)
			g1 = dgl.graph((u,v))   #make graph with local id
			g11 = dgl.graph((u,v)) 
			print("g11", sys.getsizeof(g11))
			out_degrees=g1.out_degrees()
			#print("out_degrees", out_degrees, len(out_degrees),len(self.layer_block.srcdata[dgl.NID]),len(u))
			src_out_degrees = out_degrees[len_out:]
			#print("src_out_degrees", src_out_degrees, len(src_out_degrees))
			in_degrees=g1.in_degrees()
			#print("in_degrees", in_degrees, len(in_degrees),len(self.layer_block.dstdata[dgl.NID]),len(v))
			#* find the high degree of src nodes
			avg_degree=torch.mean(src_out_degrees.float())
			src_out_degrees[src_out_degrees<avg_degree]=0  #
			high_degree_idx=torch.nonzero(src_out_degrees==0).squeeze()  #index
			#high_degree_idx = [i+len_out for i in high_degree_idx]
			high_degree_nodes = high_degree_idx.add(len_out)  # the node id same as nodes idx
			#print("g1.nodes()",g1.nodes(),high_degree_idx)
			#high_degree_nodes=g1.nodes()[high_degree_idx]          #nodes id
			#print("high_degree_nodes", high_degree_nodes, len(high_degree_nodes))
			
			g1.remove_nodes(high_degree_nodes,store_ids=True)   #keep the id of g1
			#print("remove high after:", g1.ndata[dgl.NID],len(g1.ndata[dgl.NID]))
			g1_in_degrees=g1.in_degrees()
			#print("g1_in_degrees", g1_in_degrees, len(g1_in_degrees))
			outnode_in_degrees=g1_in_degrees[0:len_out]
			#* remove the zero in degree nodes of dst
			zero_in_degrees=np.argwhere(outnode_in_degrees.numpy()==0).flatten()
			print("outnode_in_degrees", outnode_in_degrees, zero_in_degrees, len(zero_in_degrees))

			all_remove_nodes=sorted(np.append(zero_in_degrees,high_degree_nodes))
			print("all_remove_nodes", len(all_remove_nodes))
			g11.remove_nodes(all_remove_nodes,store_ids=True) 
			print("remove zero after:",g11.ndata[dgl.NID],len(g11.ndata[dgl.NID]))

			#* convert to symmetry graph 
			g2 = dgl.add_reverse_edges(g11)
			print("reverse after:", g2.ndata[dgl.NID],len(g2.ndata[dgl.NID]))
			
			#g3 = dgl.remove_self_loop(g2)
			#* metis partitioin
			tp1 = time.time()
			print("num_part", self.args.num_batch)
			get_memory('before Metis partition \n')
			partition = dgl.metis_partition(g=g2,k=self.args.num_batch)
			get_memory('after Metis partition \n')
			tp2 = time.time()
			first_out=[]
			all_out_len=0
			first_out_idx=[]
			for pid in partition:
				#print("part_graph", partition[pid])
				nids = partition[pid].ndata[dgl.NID].tolist()  #include src nodes
				#print("partition", nids)
				nids=sorted(nids)
				orig_nid = g2.ndata[dgl.NID][nids]
				#* exclude the src nodes, keep only the output nodes
				bout_nodes, indice1, indice2 = numpy.intersect1d(out_nodes, orig_nid,assume_unique=True,return_indices=True)
				#print("intersec", bout_nodes, indice1)
				all_out_len+=len(bout_nodes)
				print("bout_nodes", len(bout_nodes),all_out_len)
				first_out.append(bout_nodes.tolist())
				#first_out_idx=first_out_idx+indice1.tolist()  #concat the orig outnodes
				print("sorted(nids)",len(sorted(nids)))

			#first_out_indx=sorted(first_out_idx)
			#print("firstout_index", len(first_out_indx))
			#remain_node_idx=numpy.delete(np.arange(len_out), first_out_indx)  #remove the already assigned
			#print("remain_node_idx", remain_node_idx)
			#* deal with the remaining nodes
			remain_node_idx=zero_in_degrees
			numpy.random.shuffle(remain_node_idx)
			print("after_shuffle",len(remain_node_idx))

			start=0
			finall_out_len=0
			res=[]
			#* assign the remaining nodes to each micro-batch, if the outlength short than the avg
			for i in range(self.num_batch):
				len1=len(first_out[i])
				#print("first_len", len1)
				# if i==self.num_batch-1:
				# 	blen_out = len_out-blen_out*(self.num_batch-1)

				if len1<blen_out:      #
					need_nodes_num=blen_out-len1

					end=start+need_nodes_num    #
					if start>=len(remain_node_idx):
						pass
					if end>=len(remain_node_idx):
						need_nodes_idx=remain_node_idx[start:]
					else:
						need_nodes_idx=remain_node_idx[start:end]
					need_nodes=out_nodes[need_nodes_idx].tolist()
					#print("need nodes", len(need_nodes))
					first_out[i] = first_out[i]+need_nodes   #
					start=end

				finall_out_len+=len(first_out[i])
				print("the output", len(first_out[i]))
				
				res.append(first_out[i])
			print("finall_out_len",finall_out_len)
			self.local_batched_seeds_list=res	
			print('local_batched_seeds_list', len(res), type(res))

			print('topx partition end ----................................')
			# print('the time spent: ', tp2-ts)
			print('REG construction  time spent: ', tp1-ts)
			print('pure dgl.metis_partition the time spent: ', tp2-tp1)
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
		#! ??
		self.local_to_global() # local to global         self.global_batched_seeds_list
		t_total=time.time()-ts

		return self.global_batched_seeds_list, self.weights_list, t_total, self.partition_len_list