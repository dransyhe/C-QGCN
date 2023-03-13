"""Graph decoders."""
import manifolds
import torch.nn as nn
import torch.nn.functional as F
import torch 

from layers.att_layers import GraphAttentionLayer
from layers.layers import GraphConvolution, Linear
from layers.hyp_layers import HyperbolicGraphConvolution
from utils.distortions import compute_r_score


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, manifold,c, args):
        super(GCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)
        self.decode_adj = True

class HGCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, manifold, c, args):
        super(HGCNDecoder, self).__init__(c)
        act = lambda x: x
        # manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg
        self.cls = HyperbolicGraphConvolution(manifold, args.dim, args.n_classes, self.c, self.c, args.dropout, act, args.bias, args.use_att, args.local_agg)
        self.decode_adj = True


class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, manifold,c, args):
        super(GATDecoder, self).__init__(c)
        self.cls = GraphAttentionLayer(args.dim, args.n_classes, args.dropout, F.elu, args.alpha, 1, True)
        self.decode_adj = True


class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, manifold,c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, self.c), self.c)
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )


class MDDecoder(Decoder):
    """
    Graph Reconstruction Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, manifold, args):
        super(Decoder, self).__init__()
        self.manifold=manifold
        self.manifold_name = manifold.name
        self.input_dim = args.dim
        self.decode_adj = False
        self.beta = c
        self.curv_alpha = args.curv_alpha
        self.curv_coef = args.curv_coef

    def decode(self, x, data):
        # unpack x, r
        x, r = x[:, :-1], x[:, -1:]

        num, dim = x.size(0),x.size(1)
        device = x.get_device()
        if device >= 0:
            adj = torch.Tensor(data['adj_train'].A).to(device)
        else:
            adj = torch.Tensor(data['adj_train'].A)
        positive = adj.bool()
        negative = ~positive

        x_1 = x.repeat(num,1)
        x_2 = x.repeat_interleave(num,0)

        # prepare r_1, r_2
        r_1 = r.repeat(num, 1)
        r_2 = r.repeat_interleave(num, 0)

        # add r_dist = (r_i - r_j)^2
        r_dist = (r_1 - r_2).pow(2)

        dist = self.manifold.sqdist(x_1, x_2, self.beta).view(num,num)
        dist = dist + r_dist
        inner = self.manifold.inner(x_1, x_2).view(num,num)

        simi = torch.clamp(torch.exp(-dist), min=1e-15)
        positive_sim = simi * (positive.long())
        negative_sim = simi * (negative.long())
        
        edge_inner = inner*adj.bool()
        max_inner, min_inner = edge_inner.max().item(),edge_inner.min().item()
        
        negative_sum = negative_sim.sum(dim=1).unsqueeze(1).repeat(1,num)
        dist_loss = torch.clamp(torch.div(positive_sim, negative_sum)[positive],min=1e-15)
        # print("loss:",loss.min().item())
        dist_loss = (-torch.log(dist_loss)).sum()

        # add curv_loss
        # TODO: R_h is now omitted
        # curv_loss = sum((F(x_i) - R_h -  R(ri))^2 / (|F(x_i)| + eps))
        r_score = compute_r_score(r, self.curv_alpha)
        curv_loss = torch.sum(torch.pow((data['f_score'] - r_score), 2) /
                              torch.pow((torch.abs(data['f_score']) + 1e-15), 2))

        loss = dist_loss + self.curv_coef * curv_loss

        return x, dist, loss, dist.max().item(), max_inner, min_inner

    def extra_repr(self):
        return None

model2decoder = {
    'GCN': GCNDecoder,
    'GAT': GATDecoder,
    'HNN': LinearDecoder,
    'HGCN': LinearDecoder,
    'MLP': LinearDecoder,
    'Shallow': LinearDecoder,
}

