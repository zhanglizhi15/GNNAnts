#!/bin/bash

File=gen_data.py
#num-epochs=180

# data=ogbn-products
# data=cora
# data=pubmed
# data=reddit

data=ogbn-arxiv
file_path=../../dataset/gendata/multi_layers_full_graph/ogbn-arxiv
num_epoch=5



fan_out=10,25
if [ -d "$file_path/fan_out_10,25" ]; then
    echo "$file_path/fan_out_10,25 is exist"
else
    mkdir -p $file_path/fan_out_10,25
    python $File --fan-out=$fan_out --num-layers=2 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data
fi
#--------------------------------------------------------------------------------------------------------

fan_out=10,25,30
if [ -d "$file_path/fan_out_10,25,30" ]; then
    echo "$file_path/fan_out_10,25,30 is exist"
else
    mkdir -p $file_path/fan_out_10,25,30
    python $File --fan-out=$fan_out --num-layers=3 --num-epochs=180 --num-hidden=1 --dataset=$data
fi
#--------------------------------------------------------------------------------------------------------

fan_out=10,25,30,40
if [ -d "$file_path/fan_out_10,25,30,40" ]; then
    echo "$file_path/fan_out_10,25,30,40 is exist"
else
    mkdir -p $file_path/fan_out_10,25,30,40
    python $File --fan-out=$fan_out --num-layers=4 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data
fi
#--------------------------------------------------------------------------------------------------------

fan_out=10,25,30,40,50
if [ -d "$file_path/fan_out_10,25,30,40,50" ]; then
    echo "$file_path/fan_out_10,25,30,40,50 is exist"
else
    mkdir -p $file_path/fan_out_10,25,30,40,50
    python $File --fan-out=$fan_out --num-layers=5 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data
fi
#--------------------------------------------------------------------------------------------------------

<< EOF
data=ogbn-products
num_epoch=5
file_path=../../dataset/gendata/multi_layers_full_graph/ogbn-products

fan_out=10 #10
if [ -d "$file_path/fan_out_10" ]; then
    echo "$file_path/fan_out_10 is exist"
else
    mkdir -p $file_path/fan_out_10
    python $File --fan-out=$fan_out --num-layers=1 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data --gen-full-batch=True
fi



fan_out=10,25 #10,25
if [ -d "$file_path/fan_out_10,25" ]; then
    echo "$file_path/fan_out_10,25 is exist"
else
    mkdir -p $file_path/fan_out_10,25
    python $File --fan-out=$fan_out --num-layers=2 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data --gen-full-batch=True
fi



fan_out=10,25,30 #10,25,30
if [ -d "$file_path/fan_out_10,25,30" ]; then
    echo "$file_path/fan_out_10,25,30 is exist"
else
    mkdir -p $file_path/fan_out_10,25,30
    python $File --fan-out=$fan_out --num-layers=3 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data --gen-full-batch=True

fi


fan_out=10,25,30,40 #10,25,30,40
if [ -d "$file_path/fan_out_10,25,30,40" ]; then
    echo "$file_path/fan_out_10,25,30,40 is exist"
else
    mkdir -p $file_path/fan_out_10,25,30,40
    python $File --fan-out=$fan_out --num-layers=4 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data --gen-full-batch=True
fi



fan_out=10,25,30,40,50
if [ -d "$file_path/fan_out_10,25,30,40,50" ]; then
    echo "$file_path/fan_out_10,25,30,40,50 is exist"
else
    mkdir -p $file_path/fan_out_10,25,30,40,50
    python $File --fan-out=$fan_out --num-layers=5 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data --gen-full-batch=True
fi



data=ogbn-papers100M
num_epoch=5
file_path=../../dataset/gendata/multi_layers_full_graph/ogbn-papers100M


# fan_out=10 #10
# if [ -d "$file_path/fan_out_10" ]; then
#     echo "$file_path/fan_out_10 is exist"
# else
#     mkdir -p $file_path/fan_out_10
#     python $File --fan-out=$fan_out --num-layers=1 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data --gen-full-batch=True
# fi



fan_out=10,25 #10,25
if [ -d "$file_path/fan_out_10,25" ]; then
    echo "$file_path/fan_out_10,25 is exist"
else
    mkdir -p $file_path/fan_out_10,25
    python $File --fan-out=$fan_out --num-layers=2 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data --gen-full-batch=True
fi



fan_out=10,25,30 #10,25,30
if [ -d "$file_path/fan_out_10,25,30" ]; then
    echo "$file_path/fan_out_10,25,30 is exist"
else
    mkdir -p $file_path/fan_out_10,25,30
    python $File --fan-out=$fan_out --num-layers=3 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data --gen-full-batch=True

fi

fan_out=10,25,30,40 #10,25,30
if [ -d "$file_path/fan_out_10,25,30,40" ]; then
    echo "$file_path/fan_out_10,25,30,40 is exist"
else
    mkdir -p $file_path/fan_out_10,25,30,40
    python $File --fan-out=$fan_out --num-layers=4 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data --gen-full-batch=True
fi
EOF

# data=reddit
# file_path=../../dataset/gendata/multi_layers_full_graph/reddit
# num_epoch=5


# fan_out=10
# if [ -d "$file_path/fan_out_10" ]; then
#     echo "$file_path/fan_out_10 is exist"
# else
#     mkdir -p $file_path/fan_out_10
#     python $File --fan-out=$fan_out --num-layers=1 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data
# fi


# fan_out=10,25
# if [ -d "$file_path/fan_out_10,25" ]; then
#     echo "$file_path/fan_out_10,25 is exist"
# else
#     mkdir -p $file_path/fan_out_10,25
#     python $File --fan-out=$fan_out --num-layers=2 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data
# fi
# #--------------------------------------------------------------------------------------------------------

# fan_out=10,25,30
# if [ -d "$file_path/fan_out_10,25,30" ]; then
#     echo "$file_path/fan_out_10,25,30 is exist"
# else
#     mkdir -p $file_path/fan_out_10,25,30
#     python $File --fan-out=$fan_out --num-layers=3 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data
# fi
# #--------------------------------------------------------------------------------------------------------

# fan_out=10,25,30,40
# if [ -d "$file_path/fan_out_10,25,30,40" ]; then
#     echo "$file_path/fan_out_10,25,30,40 is exist"
# else
#     mkdir -p $file_path/fan_out_10,25,30,40
#     python $File --fan-out=$fan_out --num-layers=4 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data
# fi
# #--------------------------------------------------------------------------------------------------------

# fan_out=10,25,30,40,50
# if [ -d "$file_path/fan_out_10,25,30,40,50" ]; then
#     echo "$file_path/fan_out_10,25,30,40,50 is exist"
# else
#     mkdir -p $file_path/fan_out_10,25,30,40,50
#     python $File --fan-out=$fan_out --num-layers=5 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data
# fi
