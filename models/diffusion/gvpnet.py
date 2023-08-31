import torch 
from ..encoders.gvp import GVCEncoder
from ..encoders.ga import GAEncoder
import torch.nn as nn
from datasets.molecule.constants import num_fg_types, num_atom_types
from datasets.protein.constants import num_aa_types
num_total_type = num_fg_types + num_aa_types + num_atom_types

class GVPNet(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim=0, num_layers=6, type_out_num=num_total_type, encoder_opt={}, protein_contex=False):
        super().__init__()
        self.current_type_embedding = nn.Embedding(num_total_type + 1, node_feat_dim) 
        if protein_contex:
            self.node_feat_mixer = nn.Sequential(
                nn.Linear(node_feat_dim * 2 + 3, node_feat_dim), nn.ReLU(),
                nn.Linear(node_feat_dim, node_feat_dim),
            )
        else:
            self.node_feat_mixer = nn.Sequential(
                nn.Linear(node_feat_dim + 3, node_feat_dim), nn.ReLU(),
                nn.Linear(node_feat_dim, node_feat_dim),
            )
        self.encoder = GVCEncoder(node_feat_dim, pair_feat_dim, num_layers, **encoder_opt)

        # self.eps_crd_net = EquivLayer(
        #     nn.Linear(node_feat_dim+3, node_feat_dim), nn.ReLU(),
        #     nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
        #    nn.Linear(node_feat_dim, 3)
        # )

        self.eps_type_net = nn.Sequential(
            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
            nn.Linear(node_feat_dim, type_out_num)
        )
    
    def forward(self, p_t, s_t, beta, mask_generate, mask_sample, node_feat=None, pair_feat=None):
        """
        Args:
            p_t:    (N, L, 3).
            s_t:    (N, L).
            node_feat:   (N, L, node_dim).
            pair_feat:  (N, L, L, pair_dim).
            beta:   (N,).
            mask_generate:    (N, L).
            mask_sample:       (N, L).
        Returns:
            eps_pos: (N, L, 3).
        """
        N, L = mask_sample.size()

        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].expand(N, L, 3)
        
        if node_feat is not None:
            node_feat = self.node_feat_mixer(torch.cat([node_feat, self.current_type_embedding(s_t), t_embed], dim=-1)) # [Important] Incorporate sequence at the current step.
        else:
            node_feat = self.node_feat_mixer(torch.cat([self.current_type_embedding(s_t), t_embed], dim=-1)) # [Important] Incorporate sequence at the current step.

        node_feat, eps_pos = self.encoder(p_t, node_feat, pair_feat, mask_sample)
        # Position changes
        eps_pos = torch.where(mask_generate[:, :, None].expand_as(eps_pos), eps_pos, torch.zeros_like(eps_pos))

        # New sequence categorical distributions
        c_denoised = self.eps_type_net(node_feat)  # Un-softmax-ed, (N, L, out_type)

        return eps_pos, c_denoised