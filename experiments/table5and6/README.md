# Table 5: Peak CPU memory consumption

To denote the reduction of memory consumption between GNNAnts and other methods.
The hyperparameter is as follows: 3-layer GraphSAGE + mean aggregator with 16 batches on arxiv, reddit, products and papers100m datasets.

After `./test_mini.sh; ./test_betty.sh; ./test_gnnants.sh` you can get the results in folder `log/`

The 'VmRSS' in the results denotes the real RAM consumption.

# Table 6: The increased CPU memory consumtion during partition

The increased memory is between 'full-batch dataloader' and 'after Metis partition' in the output results.
