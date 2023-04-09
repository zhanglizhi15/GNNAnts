import torch
import torch.nn as nn
import dgl
from dgl.base import dgl_warning
import numpy as np

"""Torch modules for graph convolutions(GCN)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch.nn import init

# from .... import function as fn
# from ....base import DGLError
# from ....utils import expand_as_pair
# from ....transforms import reverse
# from ....convert import block_to_graph
# from ....heterograph import DGLBlock
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from dgl.convert import block_to_graph
from dgl.heterograph import DGLBlock
from dgl.transform import reverse

class EdgeWeightNorm(nn.Module):
    r"""

    Description
    -----------
    This module normalizes positive scalar edge weights on a graph
    following the form in `GCN <https://arxiv.org/abs/1609.02907>`__.

    Mathematically, setting ``norm='both'`` yields the following normalization term:

    .. math::
      c_{ji} = (\sqrt{\sum_{k\in\mathcal{N}(j)}e_{jk}}\sqrt{\sum_{k\in\mathcal{N}(i)}e_{ki}})

    And, setting ``norm='right'`` yields the following normalization term:

    .. math::
      c_{ji} = (\sum_{k\in\mathcal{N}(i)}e_{ki})

    where :math:`e_{ji}` is the scalar weight on the edge from node :math:`j` to node :math:`i`.

    The module returns the normalized weight :math:`e_{ji} / c_{ji}`.

    Parameters
    ----------
    norm : str, optional
        The normalizer as specified above. Default is `'both'`.
    eps : float, optional
        A small offset value in the denominator. Default is 0.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import EdgeWeightNorm, GraphConv

    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> edge_weight = th.tensor([0.5, 0.6, 0.4, 0.7, 0.9, 0.1, 1, 1, 1, 1, 1, 1])
    >>> norm = EdgeWeightNorm(norm='both')
    >>> norm_edge_weight = norm(g, edge_weight)
    >>> conv = GraphConv(10, 2, norm='none', weight=True, bias=True)
    >>> res = conv(g, feat, edge_weight=norm_edge_weight)
    >>> print(res)
    tensor([[-1.1849, -0.7525],
            [-1.3514, -0.8582],
            [-1.2384, -0.7865],
            [-1.9949, -1.2669],
            [-1.3658, -0.8674],
            [-0.8323, -0.5286]], grad_fn=<AddBackward0>)
    """
    def __init__(self, norm='both', eps=0.):
        super(EdgeWeightNorm, self).__init__()
        self._norm = norm
        self._eps = eps

    def forward(self, graph, edge_weight):
        r"""

        Description
        -----------
        Compute normalized edge weight for the GCN model.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        edge_weight : torch.Tensor
            Unnormalized scalar weights on the edges.
            The shape is expected to be :math:`(|E|)`.

        Returns
        -------
        torch.Tensor
            The normalized edge weight.

        Raises
        ------
        DGLError
            Case 1:
            The edge weight is multi-dimensional. Currently this module
            only supports a scalar weight on each edge.

            Case 2:
            The edge weight has non-positive values with ``norm='both'``.
            This will trigger square root and division by a non-positive number.
        """
        with graph.local_scope():
            if isinstance(graph, DGLBlock):
                graph = block_to_graph(graph)
            if len(edge_weight.shape) > 1:
                raise DGLError('Currently the normalization is only defined '
                               'on scalar edge weight. Please customize the '
                               'normalization for your high-dimensional weights.')
            if self._norm == 'both' and th.any(edge_weight <= 0).item():
                raise DGLError('Non-positive edge weight detected with `norm="both"`. '
                               'This leads to square root of zero or negative values.')

            dev = graph.device
            graph.srcdata['_src_out_w'] = th.ones((graph.number_of_src_nodes())).float().to(dev)
            graph.dstdata['_dst_in_w'] = th.ones((graph.number_of_dst_nodes())).float().to(dev)
            graph.edata['_edge_w'] = edge_weight

            if self._norm == 'both':
                reversed_g = reverse(graph)
                reversed_g.edata['_edge_w'] = edge_weight
                reversed_g.update_all(fn.copy_edge('_edge_w', 'm'), fn.sum('m', 'out_weight'))
                degs = reversed_g.dstdata['out_weight'] + self._eps
                norm = th.pow(degs, -0.5)
                graph.srcdata['_src_out_w'] = norm

            if self._norm != 'none':
                graph.update_all(fn.copy_edge('_edge_w', 'm'), fn.sum('m', 'in_weight'))
                degs = graph.dstdata['in_weight'] + self._eps
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                graph.dstdata['_dst_in_w'] = norm

            graph.apply_edges(lambda e: {'_norm_edge_weights': e.src['_src_out_w'] * \
                                                               e.dst['_dst_in_w'] * \
                                                               e.data['_edge_w']})
            return graph.edata['_norm_edge_weights']


class GraphConvCacheReuse(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GraphConvCacheReuse, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        print("in_feats", in_feats, out_feats)
        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, prev_layer_repeat, step, reuse_embedding, weight=None, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')


            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            #! the last layer not only for cache for the next batch but also reuse for next layer
            #! add cache embedding to src feature
            cache_embedding=torch.tensor([], dtype=torch.float32, device=graph.device) #.to(graph.device)
            reuse_indices = prev_layer_repeat[step][1] 
            if len(reuse_indices)>0:  # some nodes need reuse
                full_dst_len = rst.size(0)+len(reuse_indices)
                full_dst_indices = np.arange(full_dst_len) 
                #print("len_dst", len(full_dst_indices), len(reuse_indices), rst.size())
                unprune_indices = np.delete(full_dst_indices, reuse_indices)
                #tic = time.time()
                #full_dst_feat=temp_emb[full_dst_indices]
                full_dst_feat=torch.full((full_dst_len, rst.size(1)), 0, dtype=torch.float32, device=graph.device)
                #full_dst_feat=torch.tensor(full_dst_len, rst.size(1), device=graph.device) #.to(graph.device) #
                #tic1 = time.time()
                #print(f"tensor_cre: {tic1-tic}")
                full_dst_feat[unprune_indices] = rst  #avoid the none cannot assigment
                #if reuse_embedding.size(0)>1: 
                #print("reuse_size", step, len(reuse_indices), reuse_embedding.size())
                full_dst_feat[reuse_indices] = reuse_embedding
                rst = full_dst_feat
                #print("rst.size", rst.size())

            try:
                #//dst_nids = graph.dstdata[dgl.NID] the dst is incomplete
                cache_indices = prev_layer_repeat[step+1][0]  #* cache indice in current batch 
                ##/cache_nodes = dst_nids[cache_indices]  #cache for the dst
                #print("cache_nodes", len(cache_indices), rst.size())
                #if len(cache_indices)>0:
                cache_embedding = rst[cache_indices]
                cache_embedding = cache_embedding.detach() 
                #print("cache_embedding",cache_embedding.size())
            except:
                pass
            #print("cache_size", cache_embedding.size())
            #print("featshape",self.fc_self(h_self).size(),h_neigh.size(),rst.size())
            # see_memory_usage("----------------------------------------after rst")

            return rst, cache_embedding

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)


# pylint: disable=W0235
class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GraphConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        print("in_feats", in_feats, out_feats)
        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)



class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers,
                 activation, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        if self.n_layers==1:  #should cache the features
            self.layers.append(GraphConvCacheReuse(in_feats, n_classes,weight=True, activation=activation,allow_zero_in_degree=True))
        #self.bns.append(nn.BatchNorm1d(n_hidden))
        elif self.n_layers==2:
            self.layers.append(GraphConvCacheReuse(in_feats,  n_hidden, weight=True,activation=activation,allow_zero_in_degree=True))
                #self.bns.append(nn.BatchNorm1d(n_hidden))
            self.layers.append(GraphConv(n_hidden, n_classes,weight=True,allow_zero_in_degree=True)) #activation=activation))
        else:
            self.layers.append(GraphConv(in_feats, n_hidden,weight=True, activation=activation,allow_zero_in_degree=True))
            for i in range(1,n_layers - 2):
                self.layers.append(GraphConv(n_hidden,  n_hidden, weight=True,activation=activation,allow_zero_in_degree=True))
                #self.bns.append(nn.BatchNorm1d(n_hidden))
            self.layers.append(GraphConvCacheReuse(n_hidden,  n_hidden, weight=True,activation=activation,allow_zero_in_degree=True))
            self.layers.append(GraphConv(n_hidden, n_classes,weight=True,allow_zero_in_degree=True)) #activation=activation))
        self.dropout = nn.Dropout(p=dropout)
        #self.activation = activation

    def forward(self, blocks_and_repeat, input_reuse):
        blocks = blocks_and_repeat[0]
        prev_layer_repeat = blocks_and_repeat[1]
        step = blocks_and_repeat[2]
        x=input_reuse[0]
        reuse_embedding = input_reuse[1]
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            if l != 0:
                x = self.dropout(x)
            else:
                pass
            if (l == self.n_layers-2):  
                x, cache_embedding = layer(block, x, prev_layer_repeat, step, reuse_embedding)
            else:
                x = layer(block, x)

        return x, cache_embedding

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
