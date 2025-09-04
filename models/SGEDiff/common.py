import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_geometric.data import Batch, Data

class GaussianSmearing(nn.Module):  # default 20
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        # offset = torch.linspace(start, stop, num_gaussians)
        # customized offset

        offset = torch.tensor([0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10])
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def __repr__(self):
        return f'GaussianSmearing(start={self.start}, stop={self.stop}, num_gaussians={self.num_gaussians})'

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AngleExpansion(nn.Module):
    def __init__(self, start=1.0, stop=5.0, half_expansion=10):
        super(AngleExpansion, self).__init__()
        l_mul = 1. / torch.linspace(stop, start, half_expansion)
        r_mul = torch.linspace(start, stop, half_expansion)
        coeff = torch.cat([l_mul, r_mul], dim=-1)
        self.register_buffer('coeff', coeff)

    def forward(self, angle):
        return torch.cos(angle.view(-1, 1) * self.coeff.view(1, -1))


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    'silu': nn.SiLU()
}


class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def outer_product(*vectors):  
    for index, vector in enumerate(vectors): 
        if index == 0: 
            out = vector.unsqueeze(-1) 
        else: 
            out = out * vector.unsqueeze(1) # [edge_n, 1, 20] -> edge_n features, 4, 20
            out = out.view(out.shape[0], -1).unsqueeze(-1)
    return out.squeeze()


def get_h_dist(dist_metric, hi, hj):
    if dist_metric == 'euclidean':
        h_dist = torch.sum((hi - hj) ** 2, -1, keepdim=True)
        return h_dist
    elif dist_metric == 'cos_sim':
        hi_norm = torch.norm(hi, p=2, dim=-1, keepdim=True)
        hj_norm = torch.norm(hj, p=2, dim=-1, keepdim=True)
        h_dist = torch.sum(hi * hj, -1, keepdim=True) / (hi_norm * hj_norm)
        return h_dist, hj_norm


def get_r_feat(r, r_exp_func, node_type=None, edge_index=None, mode='basic'):
    if mode == 'origin':
        r_feat = r
    elif mode == 'basic':
        r_feat = r_exp_func(r)
    elif mode == 'sparse':
        src, dst = edge_index
        nt_src = node_type[src]  # [n_edges, 8]
        nt_dst = node_type[dst]
        r_exp = r_exp_func(r)
        r_feat = outer_product(nt_src, nt_dst, r_exp)
    else:
        raise ValueError(mode)
    return r_feat


# def protein_map(idx, h_protein, pos_protein, batch_protein):
#     act_idx = [Data(x=torch.from_numpy(i[0])) for i in idx]
#     batch_ = Batch.from_data_list(act_idx)
#     x_ = batch_.x
#
#     idx_ = x_ >= 0
#     x_ = x_[idx_]
#     batch_ = batch_.batch[idx_]
#     return h_protein[x_], pos_protein, batch_


def compose_context_single(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):  
    # previous version has problems when ligand atom types are fixed
    # (due to sorting randomly in case of same element)

    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)  
    # sort_idx = batch_ctx.argsort()
    sort_idx = torch.sort(batch_ctx, stable=True).indices  

    mask_ligand = torch.cat([
        torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]  

    batch_ctx = batch_ctx[sort_idx]  
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]  
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  

    return h_ctx, pos_ctx, batch_ctx, mask_ligand  


def compose_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand, subgraph_info):  
    batch_n = int(max(batch_protein) + 1)
    # print(imp_info["imp_p_idx"], imp_info["imp_interaction"], imp_info["imp_cls"])


    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)  
    # sort_idx = batch_ctx.argsort()
    sort_idx = torch.sort(batch_ctx, stable=True).indices  

    mask_ligand = torch.cat([
        torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]  

    batch_ctx = batch_ctx[sort_idx]  

    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]  
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  


    if subgraph_info is None:
        return (h_ctx, pos_ctx, batch_ctx, mask_ligand, h_ctx, pos_ctx, batch_ctx,
         torch.tensor([]), torch.tensor([]), mask_ligand)
    imp_batch = subgraph_info["subgraph_pidx"]  # protein + ligand  idx_pair
    imp_act = subgraph_info["subgraph_act"]  # bond_idx
    imp_cls = subgraph_info["subgraph_cls"]  # bond_cls
    imp_mask = subgraph_info["subgraph_ligand_mask"]   # protein + ligand

    # print(imp_batch)
    # print(imp_batch, imp_act, imp_cls, imp_mask)
    # for i,j in zip(imp_batch, imp_mask):
    #     print(i.shape, j.shape)
    batch_imp = torch.tensor([]).to(pos_ctx.device)
    h_imp = torch.tensor([]).to(pos_ctx.device)
    pos_imp = torch.tensor([]).to(pos_ctx.device)
    imp_bonds = torch.tensor([]).to(pos_ctx.device)
    f_imp_cls = torch.tensor([]).to(pos_ctx.device)
    imp_ligand_mask = torch.tensor([]).to(pos_ctx.device)

    # print("all", len(imp_batch[0]) + len(imp_batch[1]), len(imp_batch[0]), len(imp_batch[1]))
    # print("Interaction", len(imp_act[0]), len(imp_act[1]))

    for i in range(batch_n):
        batch_imp = torch.cat([batch_imp, torch.ones([len(imp_batch[i])]).to(pos_ctx)*i], dim=0)  
        h_imp = torch.cat([h_imp, h_ctx[batch_ctx==i][imp_batch[i]]], dim=0)  
        pos_imp = torch.cat([pos_imp, pos_ctx[batch_ctx==i][imp_batch[i]]], dim=0) 
        imp_ligand_mask = torch.cat([imp_ligand_mask, imp_mask[i]], dim=-1)
        # print(imp_mask[i].shape)
        # print()

    fix_idx = 0

    # print(imp_act, imp_cls)
    # exit()


    for i in range(batch_n):
        
        imp_bonds = torch.cat([imp_bonds, imp_act[i] + fix_idx], dim=0)
        f_imp_cls = torch.cat([f_imp_cls, imp_cls[i]], dim=-1)
        fix_idx += len(batch_imp[batch_imp == i])

    if len(f_imp_cls) > 0:  
        reverse_bond = torch.cat([imp_bonds[:, 1].unsqueeze(-1), imp_bonds[:, 0].unsqueeze(-1)], dim=-1)
        imp_bonds = torch.cat([imp_bonds, reverse_bond], dim=0)
        f_imp_cls = torch.cat([f_imp_cls, f_imp_cls], dim=-1).long()

    return (h_ctx, pos_ctx, batch_ctx, mask_ligand, h_imp, pos_imp, batch_imp,
            imp_bonds.T, f_imp_cls, imp_ligand_mask)  

class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


def hybrid_edge_connection(ligand_pos, protein_pos, k, ligand_index, protein_index):
    # fully-connected for ligand atoms
    dst = torch.repeat_interleave(ligand_index, len(ligand_index))
    src = ligand_index.repeat(len(ligand_index))
    mask = dst != src
    dst, src = dst[mask], src[mask]
    ll_edge_index = torch.stack([src, dst])

    # knn for ligand-protein edges
    ligand_protein_pos_dist = torch.unsqueeze(ligand_pos, 1) - torch.unsqueeze(protein_pos, 0)
    ligand_protein_pos_dist = torch.norm(ligand_protein_pos_dist, p=2, dim=-1)
    knn_p_idx = torch.topk(ligand_protein_pos_dist, k=k, largest=False, dim=1).indices
    knn_p_idx = protein_index[knn_p_idx]
    knn_l_idx = torch.unsqueeze(ligand_index, 1)
    knn_l_idx = knn_l_idx.repeat(1, k)
    pl_edge_index = torch.stack([knn_p_idx, knn_l_idx], dim=0)
    pl_edge_index = pl_edge_index.view(2, -1)
    return ll_edge_index, pl_edge_index


def batch_hybrid_edge_connection(x, k, mask_ligand, batch, add_p_index=False):
    batch_size = batch.max().item() + 1
    batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index = [], [], []
    with torch.no_grad():
        for i in range(batch_size):
            ligand_index = ((batch == i) & (mask_ligand == 1)).nonzero()[:, 0]
            protein_index = ((batch == i) & (mask_ligand == 0)).nonzero()[:, 0]
            ligand_pos, protein_pos = x[ligand_index], x[protein_index]
            ll_edge_index, pl_edge_index = hybrid_edge_connection(
                ligand_pos, protein_pos, k, ligand_index, protein_index)
            batch_ll_edge_index.append(ll_edge_index)
            batch_pl_edge_index.append(pl_edge_index)
            if add_p_index:
                all_pos = torch.cat([protein_pos, ligand_pos], 0)
                p_edge_index = knn_graph(all_pos, k=k, flow='source_to_target')
                p_edge_index = p_edge_index[:, p_edge_index[1] < len(protein_pos)]
                p_src, p_dst = p_edge_index
                all_index = torch.cat([protein_index, ligand_index], 0)
                p_edge_index = torch.stack([all_index[p_src], all_index[p_dst]], 0)
                batch_p_edge_index.append(p_edge_index)

    if add_p_index:
        edge_index = [torch.cat([ll, pl, p], -1) for ll, pl, p in zip(
            batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index)]
    else:
        edge_index = [torch.cat([ll, pl], -1) for ll, pl in zip(batch_ll_edge_index, batch_pl_edge_index)]
    edge_index = torch.cat(edge_index, -1)
    return edge_index
