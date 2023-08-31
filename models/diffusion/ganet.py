from ..encoders.ga import GAEncoder
from ..encoders.egnn import EGNNLayer
import torch.nn as nn
from datasets.molecule.constants import num_fg_types, num_atom_types
from datasets.protein.constants import num_aa_types
num_total_type = num_fg_types + num_aa_types + num_atom_types
import torch 
from ..utils.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3
from ..utils.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix

class GANet(nn.Module):

    def __init__(self, fg_feat_dim, pair_feat_dim, num_layers, type_out_num=num_total_type, use_egnn_update=True, encoder_opt={}):
        super().__init__()
        self.current_type_embedding = nn.Embedding(num_total_type + 1, fg_feat_dim) 
        self.fg_feat_mixer = nn.Sequential(
            nn.Linear(fg_feat_dim * 2, fg_feat_dim), nn.ReLU(),
            nn.Linear(fg_feat_dim, fg_feat_dim),
        )
        self.encoder = GAEncoder(fg_feat_dim, pair_feat_dim, num_layers, **encoder_opt)

        self.eps_crd_net = nn.Sequential(
            nn.Linear(fg_feat_dim+3, fg_feat_dim), nn.ReLU(),
            nn.Linear(fg_feat_dim, fg_feat_dim), nn.ReLU(),
            nn.Linear(fg_feat_dim, 3)
        )

        self.eps_rot_net = nn.Sequential(
            nn.Linear(fg_feat_dim+3, fg_feat_dim), nn.ReLU(),
            nn.Linear(fg_feat_dim, fg_feat_dim), nn.ReLU(),
            nn.Linear(fg_feat_dim, 3)
        )

        self.eps_type_net = nn.Sequential(
            nn.Linear(fg_feat_dim+3, fg_feat_dim), nn.ReLU(),
            nn.Linear(fg_feat_dim, fg_feat_dim), nn.ReLU(),
            nn.Linear(fg_feat_dim, type_out_num)
        )
        if use_egnn_update:
            self.egnn_update = EGNNLayer(fg_feat_dim)

    def forward(self, v_t, p_t, s_t, fg_feat, pair_feat, beta, mask_generate, mask_sample):
        """
        Args:
            v_t:    (N, L, 3).
            p_t:    (N, L, 3).
            s_t:    (N, L).
            fg_feat:   (N, L, res_dim).
            pair_feat:  (N, L, L, pair_dim).
            beta:   (N,).
            mask_generate:    (N, L).
            mask_sample:       (N, L).
        Returns:
            v_next: UPDATED (not epsilon) SO3-vector of orietnations, (N, L, 3).
            eps_pos: (N, L, 3).
        """
        N, L = mask_sample.size()
        R = so3vec_to_rotation(v_t) # (N, L, 3, 3)
        mask_single_atom = torch.logical_and((v_t.norm(dim=-1) <= 1e-6), mask_generate)

        fg_feat = self.fg_feat_mixer(torch.cat([fg_feat, self.current_type_embedding(s_t)], dim=-1)) # [Important] Incorporate sequence at the current step.
        fg_feat = self.encoder(R, p_t, fg_feat, pair_feat, mask_sample)

        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].expand(N, L, 3)
        in_feat = torch.cat([fg_feat, t_embed], dim=-1)

        # Position changes
        eps_crd = self.eps_crd_net(in_feat)    # (N, L, 3)
        eps_pos = apply_rotation_to_vector(R, eps_crd)  # (N, L, 3)
        eps_pos = torch.where(mask_generate[:, :, None].expand_as(eps_pos), eps_pos, torch.zeros_like(eps_pos))

        if mask_single_atom.sum() > 0:
            eps_pos_linker = self.egnn_update(p_t, in_feat, mask_sample, mask_single_atom)
            eps_pos = torch.where(mask_single_atom[:, :, None].expand_as(eps_pos), eps_pos_linker, eps_pos)
        
        # New orientation
        eps_rot = self.eps_rot_net(in_feat)    # (N, L, 3)
        U = quaternion_1ijk_to_rotation_matrix(eps_rot) # (N, L, 3, 3)
        R_next = R @ U
        v_next = rotation_to_so3vec(R_next)     # (N, L, 3)
        v_next = torch.where(mask_generate[:, :, None].expand_as(v_next), v_next, v_t)

        # New sequence categorical distributions
        c_denoised = self.eps_type_net(in_feat)  # Un-softmax-ed, (N, L, out_type)

        return v_next, R_next, eps_pos, c_denoised

