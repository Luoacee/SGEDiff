import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_softmax, scatter_sum
from models.SGEDiff.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product


# from models.egnn_new import EGNN


def build_edge_type(edge_index, mask_ligand):  
    src, dst = edge_index  
    edge_type = torch.zeros(len(src)).to(edge_index)  
    n_src = mask_ligand[src] == 1  # n_srcbool [T F T T T F] ligand
    n_dst = mask_ligand[dst] == 1  # n_dst
    edge_type[n_src & n_dst] = 0  # T，
    edge_type[n_src & ~n_dst] = 1  # ->
    edge_type[~n_src & n_dst] = 2  # ->
    edge_type[~n_src & ~n_dst] = 3  # ->
    edge_type = F.one_hot(edge_type, num_classes=4)  
    return edge_type


def connect_edge(x, mask_ligand, batch, cutoff_mode, param):  # ，，，batch
    if cutoff_mode == 'radius':  
        edge_index = radius_graph(x, r=param, batch=batch, flow='source_to_target')
    elif cutoff_mode == 'knn':  # ， -> 
        edge_index = knn_graph(x, k=param, batch=batch, flow='source_to_target')  # knn
    elif cutoff_mode == 'hybrid':  # ，，
        edge_index = batch_hybrid_edge_connection(
            x, k=param, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
    else:
        raise ValueError(f'Not supported cutoff mode: {cutoff_mode}')
    return edge_index


class BaseX2HAttLayer(nn.Module):  # X -> h 
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        # attention key func
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim  # 340  128*2+4+80
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):  # ，+（+），，
        N = h.size(0)  
        src, dst = edge_index  # -
        hi, hj = h[dst], h[src]  # ，
        # multi-head attention
        # decide inputs of k_func and v_func

        kv_input = torch.cat([r_feat, hi, hj], -1)  # ，80. 128 ，128 +

        if edge_feat is not None:  
            kv_input = torch.cat([edge_feat, kv_input], -1)  # 4 + 336    + 
        
        # compute k
        kv_input = kv_input.float()
        k = self.hk_func(kv_input).view(-1, self.n_heads,
                                        self.output_dim // self.n_heads)  # k [edges, head, //] [59616, 16, 8]

        # compute v
        v = self.hv_func(kv_input)  # [59616, 128]

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.

        v = v * e_w  # vew，
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)  # v

        # compute q  ， 
        q = self.hq_func(h).view(-1, self.n_heads,
                                 self.output_dim // self.n_heads)  # q  [edges, head, //] [59616, 16, 8]

        # compute attention weights, ，
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,  # [59616, 16] [59616]
                                dim_size=N)  # [num_edges, n_heads]
        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head) 。
        output = output.view(-1, self.output_dim)  # (N, 128)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))  

        output = output + h  # res 
        return output  


class BaseH2XAttLayer(nn.Module):  # h-> x 
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):  # ，，+，，
        N = h.size(0)
        src, dst = edge_index  
        hi, hj = h[src], h[dst]  # ，

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)  
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)  

        kv_input = kv_input.float()
        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)  
        v = self.xv_func(kv_input)
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:  
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w  
        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]  # v ？？？  

        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)  # X

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads) 
        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3) 
        return output.mean(1)  # [num_nodes, 3]


class SubGraphProcess(nn.Module):  # X -> h 
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim,
                 act_fn='relu', norm=True, out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.out_fc = out_fc

        # attention key func
        kv_input_dim = input_dim + edge_feat_dim  # 340  128*2+4+80
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, edge_feat, edge_index, e_w=None):  # ，+（+），，
        N = h.size(0)  
        src, dst = edge_index  # -
        hi, hj = h[src], h[dst]  # ，

        kv_input = hi  # ，128 ，128 +
        if edge_feat is not None:  
            kv_input = torch.cat([kv_input, edge_feat], -1)  # 4 + 336    + 
        
        # compute k

        kv_input = kv_input.float()
        k = self.hk_func(kv_input).view(-1, self.n_heads,
                                        self.output_dim // self.n_heads)  # k [edges, head, //] [59616, 16, 8]
        # compute v
        v = self.hv_func(kv_input)
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)  # v
        # compute q  ， 
        q = self.hq_func(h).view(-1, self.n_heads,
                                 self.output_dim // self.n_heads)  # q  [edges, head, //] [59616, 16, 8]
        # compute attention weights, ，
        alpha = scatter_softmax((q[dst] * k).sum(-1) / np.sqrt(k.shape[-1]), dst, dim=0,  # [59616, 16] [59616]
                                dim_size=N)  # [num_edges, n_heads]
        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head) 。
        output = output.view(-1, self.output_dim)  # (N, 128)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))  

        output = output + h  # res 
        return output  


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):  # h
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian  # 20
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h  # 1
        self.num_h2x = num_h2x  # 1
        self.r_min, self.r_max = r_min, r_max  # 0, 10
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)  # 0， 10， 20

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):  # 1 
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * self.edge_feat_dim,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):  # 1 x
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * self.edge_feat_dim,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False):  
        src, dst = edge_index  
        if self.edge_feat_dim > 0:  # 4，
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None

        rel_x = x[dst] - x[src]  
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)  # ，，[edge_n, 1]

        h_in = h
        # 4 separate distance embedding for p-p, p-l, l-p, l-l
        for i in range(self.num_x2h):  # 1
            dist_feat = self.distance_expansion(dist)  # 20 4
            dist_feat = outer_product(edge_attr, dist_feat)  # onehot， [edge_n, 80]
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)  # ，，。
            h_in = h_out  # h_
        x2h_out = h_in  # ,

        new_h = h if self.sync_twoup else x2h_out  # hx
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)  
            dist_feat = outer_product(edge_attr, dist_feat)  
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)
            if not fix_x:  # x，x，x,x
                x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated，
            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)  

        return x2h_out, x  


class SubgraphNet(nn.Module):  # h
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian  # 20
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h  # 1
        self.num_h2x = num_h2x  # 1
        self.r_min, self.r_max = r_min, r_max  # 0, 10
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):  # 1 
            self.x2h_layers.append(
                SubGraphProcess(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                act_fn=act_fn, norm=norm, out_fc=self.x2h_out_fc)
            )

    def forward(self, h, edge_attr, edge_index):  
        if self.edge_feat_dim > 0:  # 4，
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None

        h_in = h
        for i in range(self.num_x2h):  # 1
            h_out = self.x2h_layers[i](h_in, edge_feat, edge_index)  # ，，。
            h_in = h_out  # h_
        x2h_out = h_in  # ,
        return x2h_out  


class SGEDiff(nn.Module):  #  -> kernel
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, k=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='radius', ew_net_type='r', n_cross_layer=3,
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        # Build the network
        self.num_blocks = num_blocks  # 1  
        self.num_layers = num_layers  # 9  
        self.hidden_dim = hidden_dim  # 128  
        self.n_heads = n_heads  # 16  transformer
        self.num_r_gaussian = num_r_gaussian  # 20 
        self.edge_feat_dim = edge_feat_dim  # 4 
        self.act_fn = act_fn  # relu  
        self.norm = norm  # True  
        self.num_node_types = num_node_types  # 8 ？？？
        # radius graph / knn graph
        self.cutoff_mode = cutoff_mode  # [radius, none]  knn  
        self.k = k  # 32
        self.ew_net_type = ew_net_type  # [r, m, none] globe  ew

        self.num_x2h = num_x2h  # 1  
        self.num_h2x = num_h2x  # 1  
        self.num_init_x2h = num_init_x2h  # 1   x-h
        self.num_init_h2x = num_init_h2x  # 0   h-x
        self.r_max = r_max  # 10
        self.x2h_out_fc = x2h_out_fc  # False
        self.sync_twoup = sync_twoup  # False
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)  
        self.n_cross_layer = n_cross_layer

        if self.ew_net_type == 'global':  
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)  # ，，：，gaussian1

        self.init_h_emb_layer = self._build_init_h_layer()  # ，AttentionLayerO2TwoUpdateNodeGeneral 
        self.base_block = self._build_share_blocks()  # ，AttentionLayerO2TwoUpdateNodeGeneral
        # self.sub_block = self._build_sub_blocks(edge_feat_dim=4)

        self.cross_attention_block = self._build_cross_attention_block()
        self.subgraph_layer = self._build_subgraph_layer(edge_feat_dim=10)

    def _build_cross_attention_block(self):  
        base_block = []
        for l_idx in range(self.n_cross_layer):
            layer = EnrichmentNet(self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim,
                                           act_fn='relu', norm=True,
                                           num_cl=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                                           ew_net_type='r', cl_out_fc=True, sync_twoup=False)
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):  # h
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )

        return layer

    def _build_subgraph_layer(self, edge_feat_dim):  
        base_block = []
        for l_idx in range(self.num_layers):
            layer = SubgraphNet(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
                num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max,
                num_node_types=self.num_node_types,
                x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def _build_share_blocks(self):  # h，。
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(  # Attenlayers
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def maingraph_update(self, h, x, mask_ligand, batch, cut_model, param, return_all=False,
                           fix_x=False):  # h, x, ，batch
        # h x mask_ligand batch
        # all_x = [x]  # x
        # all_h = [h]  # h
        for b_idx in range(self.num_blocks):  # 1
            edge_index = connect_edge(x, mask_ligand, batch, cut_model, param)  # ,knn
            src, dst = edge_index  

            # edge type (dim: 4)
            edge_type = build_edge_type(edge_index, mask_ligand)  # ，4onehot，
            # torch.save([h, x, mask_ligand, batch, edge_index, edge_type], "sub_radius_3.pt")
            # exit()
            if self.ew_net_type == 'global':  #
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)  
                dist_feat = self.distance_expansion(dist)  # 20（）
                logits = self.edge_pred_layer(dist_feat)  # 201（）  # ！！！！！
                e_w = torch.sigmoid(logits)  # sigmoid。 ？？ 
            else:
                e_w = None

            for l_idx, layer in enumerate(self.base_block):  # ，init， id：layer
                h, x = layer(h, x, edge_type, edge_index, mask_ligand, e_w=e_w, fix_x=fix_x)  # ，

        return x, h  

    @staticmethod
    def features_bonds(edge_index, imp_edge_idx, imp_edge_cls):
        if len(imp_edge_cls) == 0:
            features_onehot = torch.zeros(size=(len(edge_index.T), 6)).to(edge_index)
            return edge_index, features_onehot

        matches = (imp_edge_idx.T[:, None, :] == edge_index.T[None, :, :]).all(dim=2)  # [1000, 10]
        # ,
        matched_imp, matched_fg = torch.where(matches)

        unmatched_edges = edge_index.T[~matched_imp]
        unmatched_features_onehot = F.one_hot(imp_edge_cls[~matched_imp], num_classes=6)

        imp_features_onehot = F.one_hot(imp_edge_cls[matched_imp], num_classes=6)  
        features_onehot = torch.zeros(size=(len(edge_index.T), 6)).to(imp_edge_cls)  # onehot
        features_onehot[matched_fg] = imp_features_onehot  

        edge_index = torch.cat([edge_index.T, unmatched_edges], dim=0)
        features_onehot = torch.cat([features_onehot, unmatched_features_onehot], dim=0)

        return edge_index.T, features_onehot


    def subgraph_update(self, h, x, mask_ligand, batch, d2_edge_idx, d2_edge_cls, cut_mode, param, return_all=False,
                        fix_x=False):

        for b_idx in range(self.num_blocks):  # 1
            edge_index = connect_edge(x, mask_ligand, batch, cut_mode, param)  # ,knn

            # edge_index, imp_edge_idx, edge_type,  imp_edge_cls
            edge_index, edge_features = self.features_bonds(edge_index, d2_edge_idx, d2_edge_cls)
            src, dst = edge_index  
            # edge type (dim: 4)
            edge_type = build_edge_type(edge_index, mask_ligand)  # ，4onehot，
            edge_type = torch.cat([edge_type, edge_features], dim=-1)

            for l_idx, layer in enumerate(self.subgraph_layer):  # ，init， id：layer
                h = layer(h, edge_type, edge_index)  # ，

        return h

    def forward(self, h, x, mask_ligand, batch, subgraph, return_all=False, fix_x=False):
        subg_h, subg_pos, subg_batch, subg_mask, subg_bonds, subg_cls = subgraph.values()
        # x_out, h_out = self.egnn(h, x, mask_ligand, batch, self.cutoff_mode, self.k)


        if len(subg_cls) > 0:
            imp_h_out = self.subgraph_update(subg_h, subg_pos, subg_mask, subg_batch, subg_bonds, subg_cls, "radius", 4)
            for i in range(len(self.cross_attention_block)):
                h = self.cross_attention_block[i](h, imp_h_out, batch, subg_batch)

        x_out, h_out = self.maingraph_update(h, x, mask_ligand, batch, self.cutoff_mode, self.k)

        outputs = {'x': x_out, 'h': h_out, "mask": mask_ligand}
        return outputs


class EnrichmentLayer(nn.Module):  # X -> h 
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.out_fc = out_fc

        # attention key func
        kv_input_dim = input_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        self.output = MLP(hidden_dim*2, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)


    def forward(self, h, sh, batch, sub_batch):  # ，+（+），，
        k = self.hk_func(sh).view(-1, self.n_heads,
                    self.output_dim // self.n_heads)
        v = self.hv_func(sh).view(-1, self.n_heads,
                    self.output_dim // self.n_heads)
        # compute q  ， 
        q = self.hq_func(h).view(-1, self.n_heads,
                                 self.output_dim // self.n_heads)  # q  [edges, head, //] [59616, 16, 8]
        q = q.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        k = k.permute(1, 2, 0)

        scores = torch.matmul(q, k) / torch.sqrt(torch.tensor(self.output_dim).to(h))
        mask = self.get_mask(batch, sub_batch).to(h)
        attention_weights = F.softmax(scores*mask, dim=-1)

        output = torch.matmul(attention_weights, v).permute(1, 0, 2).contiguous()
        output = output.view(-1, self.output_dim)
        output = self.output(torch.cat([output, h], -1))  
        return output + h

    @staticmethod
    def get_mask(batch1, batch2):
        mask = batch1.unsqueeze(1) == batch2.unsqueeze(0)
        mask = torch.where(mask, torch.tensor(1.0), torch.tensor(float(-1e9)))
        return mask


class EnrichmentNet(nn.Module):  # h
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_cl=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', cl_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian  # 20
        self.norm = norm
        self.act_fn = act_fn
        self.num_cl = num_cl  # 1
        self.num_h2x = num_h2x  # 1
        self.r_min, self.r_max = r_min, r_max  # 0, 10
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.cl_out_fc = cl_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)  # 0， 10， 20

        self.cross_layer = nn.ModuleList()

        for _ in range(self.num_cl):
            self.cross_layer.append(
                EnrichmentLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                             r_feat_dim=num_r_gaussian * 4, act_fn=act_fn, norm=norm, out_fc=self.cl_out_fc)
            )

    def forward(self, h, sh, batch, sub_batch):  

        for i in range(self.num_cl):
            h_out = self.cross_layer[i](h, sh, batch, sub_batch)  # ，，。
            h = h_out  # h_
        return h
