#!/bin/bash

File=GNNANTS_micro_batch_train.py

# Data=ogbn-products
Data=ogbn-arxiv
# Data=cora
model=sage
seed=1236 
setseed=True
GPUmem=True
load_full_batch=True
lr=0.001
dropout=0.2

run=1
epoch=180
logIndent=2
num_batch=(2 4 8)


layersList=(3)
fan_out_list=('10,25,30')

hiddenList=(128)
AggreList=(mean)

mkdir -p ./log/
savePath='./log/'

for Aggre in ${AggreList[@]}
do       
	for layers in ${layersList[@]}
	do      
		for hidden in ${hiddenList[@]}
		do
			for fan_out in ${fan_out_list[@]}
			do
				
			for nb in ${num_batch[@]}
			do
				wf=${layers}-layer-fo-${fan_out}-sage-${Aggre}-h-${hidden}-batch-${nb}.log
				echo $wf

				python $File \
				--dataset $Data \
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
				--eval \
				> ${savePath}${wf}
				done
			done
		done
	done
done
