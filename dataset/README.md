### Full graph data

The original full graph Datasets come from:
https://github.com/dglai/dgl-0.5-benchmark  and http://snap.stanford.edu/ogb/data/nodeproppred/ Node classification category.
Place the downloaded original dataset in ./origindata

### Full batch data

In this folder, we will save the generated full batch data into ./gendata/multi_layer_full_graph.
We sample from full graph based on fanout size. Each sampling setting corresponds to a folder. For example, the full batch data of **fanout 10,25** will save to folder `fan_out_10,25/`
