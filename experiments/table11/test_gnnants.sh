#!/bin/bash
#python micro_batch_train_REG.py --num-layers 3 --fan-out "10,25,30" --dataset 'ogbn-arxiv' > ogbn-arxiv-3.txt #ogbn-products #'ogbn-arxiv'
#python micro_batch_train_REG_prune.py --num-layers 3 --fan-out "10,25,30" --dataset  'ogbn-products' > ogbn-product-3-prune.txt #ogbn-products #'ogbn-arxiv'
#python micro_batch_train_REG.py --num-layers 2 --fan-out "10,25" --num-batch 2 --num-epochs 1 --aggre 'mean' --dataset  'ogbn-arxiv'

File=gnnants_micro_batch_train.py

Data=(ogbn-arxiv reddit ogbn-products) 
#  ogbn-arxiv ogbn-products reddit ogbn-papers100M
model=sage
seed=1236 
setseed=True
GPUmem=True
load_full_batch=True
lr=0.001
dropout=0.2

run=1
epoch=5
#!5
logIndent=0

num_batch=(64) 
#(2 4 8 16 32)

lenlist=1
#3
layersList=(2) 
# 3 4 5
fan_out_list=('10,25')
#'10,25'  '10,25,30' '10,25,30,40'  '10,25,30,40,50'

hiddenList=(256)
AggreList=(lstm)
# lstm
mkdir ./log
mkdir -p ./log/gnnants_micro_batch_train/
save_path='./log/gnnants_micro_batch_train/'

for Aggre in ${AggreList[@]}
do       
	for dataset in ${Data[@]}
	do 
		echo 'dataset:'${dataset}
		for ((i=0;i<$lenlist;i++))
		do  
			layers=${layersList[i]}
			echo 'layers:'${layers}
			fan_out=${fan_out_list[i]}
			echo 'fan_out:'${fan_out}
			for hidden in ${hiddenList[@]}
			do
				for nb in ${num_batch[@]}
				do
				echo 'number of batches:'${nb}
				echo ${save_path}
				wf=${save_path}${dataset}_layers_${layers}_${fan_out}_${model}_${nb}_batch_${Aggre}.log
				echo ${wf}
				python $File \
				--dataset $dataset \
				--aggre $Aggre \
				--seed $seed \
				--setseed $setseed \
				--GPUmem $GPUmem \
				--num-batch $nb \
				--lr $lr \
				--num-runs $run \
				--num-epochs $epoch \
				--num-layers $layers \
				--num-hidden $hidden \
				--dropout $dropout \
				--fan-out $fan_out \
				--log-indent $logIndent \
				--load-full-batch True \
				> ${wf}
				done
			done
		done
	done
done
