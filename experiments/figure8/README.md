# Figure 8 convergence curves

We would like to show the convergence curves for full-batch training and micro-batch training with three different numbers of batches.
In this way, it can prove the micro batch training won't change the convergence of training.
We use 3-layer GraphSAGE model + Mean aggregator using OGBN-arxiv as an example.

First, we should generate the 3-layers full-batch of 180 epoches using the code in '~/GNNANTS/pytorch/gendata/'

The pre-generated full batch data is stored in '/GNNANTS/dataset/gendata/multi_layers_full_graph/'
as we use fanout 10,25,30 these full batch data of arxiv are stored in folder '~/GNNANTS/dataset/gendata/multi_layers_full_graph/'

`./run_Betty.sh`: gets the convergency results of Betty method.

`./run_gnnants.sh`: gets the convergency results of GNNANTS method.
Then you will get the training data for full batch, 2, 4 and 8 micro batch train in folder log/.
After that, unsing the data_collection.py collect the test accuracy to draw the convergence curve.
