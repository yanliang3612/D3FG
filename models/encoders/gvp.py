import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..utils.geometry import global_to_local, local_to_global, normalize_vector, construct_3d_basis, angstrom_to_nm
from ..modules.layers import mask_zero, LayerNorm
from datasets.protein.constants import BBHeavyAtom


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out

class GVCEncoder(nn.Module):
    def __init__(self, node_feat_dim, pair_feat_dim=0, num_layers=6, ga_block_opt={}):
        super(GVCEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            GVCBlock(node_feat_dim, pair_feat_dim, **ga_block_opt) 
            for _ in range(num_layers)
        ])

    def forward(self, pos, node_feat, pair_feat, mask):
        for i, block in enumerate(self.blocks):
            node_feat, pos = block(pos, node_feat, pair_feat, mask)
        return node_feat, pos

class GVCBlock(nn.Module):
    def __init__(self, node_feat_dim, pair_feat_dim=0, node_vec_dim=1, vector_gate=True):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.node_vec_dim = node_vec_dim
        self.pair_feat_dim = pair_feat_dim

        gvp_dim_in = (2 * self.node_feat_dim + self.pair_feat_dim, 2 * self.node_vec_dim + 1)
        gvp_dim_out = (self.node_feat_dim, 1)
        self.gvp = GVP(gvp_dim_in, gvp_dim_out, vector_gate=vector_gate)
    
    def forward(self, pos, node_feat, pair_feat=None, mask=None):
        N, L, D = node_feat.shape
        mask_row = mask.view(N, L, 1, 1)  # (N, L, *, *)
        mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)

        edge_idx = mask_pair.to_sparse().indices()
        pos_concat = torch.stack(
            [pos[edge_idx[0], edge_idx[1], :],
             pos[edge_idx[0], edge_idx[2], :],
             (pos[edge_idx[0], edge_idx[1], :] - 
              pos[edge_idx[0], edge_idx[2], :])],
             dim=-2)
        
        if pair_feat is not None:
            pair_feat_concat = pair_feat[edge_idx[0], edge_idx[1], edge_idx[2]]
            node_feat_concat = torch.cat(
                [node_feat[edge_idx[0], edge_idx[1], :], 
                node_feat[edge_idx[0], edge_idx[2], :],
                pair_feat_concat],
                dim=-1)
        else:
            node_feat_concat = torch.cat(
                [node_feat[edge_idx[0], edge_idx[1], :], 
                node_feat[edge_idx[0], edge_idx[2], :]],
                dim=-1)

        node_feat_update, pos_update = self.gvp((node_feat_concat, pos_concat))
        node_feat_out = torch.zeros_like(mask_pair).float().repeat(1, 1, 1, D)
        pos_out = torch.zeros_like(mask_pair).float().repeat(1, 1, 1, 3)
        ## the same as scatter sum
        node_feat_out[edge_idx[0], edge_idx[1], edge_idx[2]] = node_feat_update
        pos_out[edge_idx[0], edge_idx[1], edge_idx[2]] = pos_update.squeeze(dim=1)

        node_num = mask.sum(dim=-1)
        node_feat_out = node_feat_out.sum(dim=-2)/node_num.view(N, 1, 1)
        pos_out = pos_out.sum(dim=-2)/node_num.view(N, 1, 1)

        return node_feat_out, pos_out


   

class GVP(nn.Module):
    '''
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.
    
    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`, 
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)    
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo: 
                v = self.wv(vh) 
                v = torch.transpose(v, -1, -2)
                if self.vector_gate: 
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return (s, v) if self.vo else s
