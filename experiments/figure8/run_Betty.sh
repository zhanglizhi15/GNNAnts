#!/bin/bash

File=Betty_micro_batch_train.py

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
# num_batch=(1 )
num_batch=(2 4 8)

pMethodList=(REG)

num_re_partition=(0)
# re_partition_method=REG
re_partition_method=random


layersList=(3)
fan_out_list=('10,25,30')

hiddenList=(128)
AggreList=(mean)

mkdir -p ./log/
savePath='./log/'

for Aggre in ${AggreList[@]}
do      
	for pMethod in ${pMethodList[@]}
	do      
		
			for layers in ${layersList[@]}
			do      
				for hidden in ${hiddenList[@]}
				do
					for fan_out in ${fan_out_list[@]}
					do
						
						for nb in ${num_batch[@]}
						do
							
							for rep in ${num_re_partition[@]}
							do
								wf=${layers}-layer-fo-${fan_out}-sage-${Aggre}-h-${hidden}-batch-${nb}-gp-${pMethod}.log
								echo $wf

								python $File \
								--dataset $Data \
								--aggre $Aggre \
								--seed $seed \
								--setseed $setseed \
								--GPUmem $GPUmem \
								--selection-method $pMethod \
								--re-partition-method $re_partition_method \
								--num-re-partition $rep \
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
		
	done
done
