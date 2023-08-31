import torch.nn as nn
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max

class EGNNLayer(nn.Module):
    def __init__(self, node_feat_dim, egnn_block_opt={}):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.project_down = nn.Linear(self.node_feat_dim + 3, 3)
    
    def forward(self, pos, node_feat, mask_sample, mask_update):
        N, L, D = node_feat.shape
        mask_row = mask_sample.view(N, L, 1, 1)  # (N, L, *, *)
        mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)

        pos_minus =  (pos.unsqueeze(dim=-2) - pos.unsqueeze(dim=-3)) * mask_pair
        node_feat_sum = (node_feat.unsqueeze(dim=-2) + node_feat.unsqueeze(dim=-3)) * mask_pair
        node_feat_project = self.project_down(node_feat_sum)

        norm = 1. / torch.square(pos_minus.norm(dim=-1) + 1.)

        pos_update = pos_minus * node_feat_project * norm.unsqueeze(-1)

        pos_update_agg = pos_update.sum(dim=1) - pos_update.sum(dim=2)

        pos_update_agg = torch.where(mask_update[:, :, None].expand_as(pos), pos_update_agg, torch.zeros_like(pos))
        
        return pos_update_agg