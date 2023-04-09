# GNNAnts: Scalable Graph Neural Networks Training with Deepest Redundant Tree Optimization

## Install requirements:

 The framework of GNNAnts is developed upon [DGL](https://github.com/dmlc/dgl) and [Betty](https://github.com/PASAUCMerced/Betty)
 We use Ubuntu 18.04, CUDA 11.1,

The package version you need to install are denoted in install_requirements.sh.
 The requirements:  pytorch >= 1.7, DGL >= 0.7, python >= 3.6

 (python 3.6 is the basic configuration in requirements here, you can use other python version, e.g. python3.8, you need configure the corresponding pytorch and dgl version.)

`bash install_requirements.sh`.

## Structure of dirctlory

- The directory **/pytorch** contains all necessary files for the GNNAnts micro-batch training, Betty micro_batch training and mini-batch training.  In folder micro_batch_train_prune,
- `graph_partitioner_topx.py` contains our implementation of topx graph partitioning.
  `block_dataloader_prune.py` is implemented to construct the pruned micro-batches based on the partitioning results of topx. `gnnants_micro_batch_train.py` modifies the code of the micro_batch training  for caching and reusing.
- The directory **/models** contains 3 commonly used models in GNN and 3 models suitable for pruned micro_batch training.
- The directory **/utils** contains the files for dataset loading and CPU, GPU memory analysis.
- You can download the [benchmarks dataset](http://snap.stanford.edu/ogb/data/nodeproppred/) into **/dataset/origindata** and generate full batch data into folder **/dataset/gendata/multi_layers_full_graph**.
- The folder **/experiments** contains these important experiment results for analysis and performance evaluation.

### The main steps for code reproduction on your own device:

- step0: Obtain the artifact, extract the archive files `git clone https://github.com/HaibaraAiChan/Betty.git`.
- step1: generate some full batch data for later experiments, (the generated data will be stored in ~/**dataset/gendata/multi_layers_full_graph**). `cd /GNNANTS/pytorch/gendata/`**./gen_data.sh**
- step2: replicate these experiments in **experiments/**
  `cd experiments/table*/` to test the experiments follow the instruction in `README.md` in corresponding figure folder.
