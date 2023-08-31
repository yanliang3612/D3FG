import torch.nn as nn
from datasets.molecule.constants import num_fg_types, num_atom_types
from datasets.protein.constants import num_aa_types
import torch 
from .gvpnet import GVPNet
from .transition import PositionTransition, PLTypeTransition
import torch.nn.functional as F
from .loss import *
import functools
from tqdm.auto import tqdm

num_total_type = num_fg_types + num_aa_types + num_atom_types

class LinkerDPM(nn.Module):

    def __init__(
        self, 
        node_feat_dim, 
        pair_feat_dim, 
        num_steps, 
        eps_net_opt={'type_out_num': num_total_type}, 
        trans_pos_opt={}, 
        trans_type_opt={
            'min_type_num':num_aa_types, 
            'max_type_num':num_aa_types + num_atom_types, 
            'num_classes': num_total_type
            },
        position_mean=[0.0, 0.0, 0.0],
        position_scale=10.0,
    ):

        super().__init__()
        self.eps_net = GVPNet(node_feat_dim, pair_feat_dim, **eps_net_opt)
        self.num_steps = num_steps
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.trans_type = PLTypeTransition(num_steps, **trans_type_opt)

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor([position_scale]).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p

    def forward(
        self, 
        p_0, 
        s_0, 
        mask_generate, 
        mask_sample, 
        linker_feat=None, 
        pair_feat=None, 
        denoise_structure=True, 
        denoise_type=True, 
        t=None
        ):

        batch_size = s_0.shape[0]

        if t is None:
            t = torch.randint(0, self.num_steps, (batch_size,), dtype=torch.long, device=self._dummy.device)
        elif len(t.shape) == 0:
            t = t.repeat(batch_size)

        p_0 = self._normalize_position(p_0)

        if denoise_structure:
            p_noisy, eps_p, pos_mask = self.trans_pos.add_noise(p_0, mask_generate, t)
        else:
            p_noisy = p_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        if denoise_type:
            # Add noise to sequence
            s_0_ignore, s_noisy, type_mask = self.trans_type.add_noise(s_0, mask_generate, t)
        else:
            s_noisy = s_0.clone()

        beta = self.trans_pos.var_sched.betas[t]
        eps_p_pred, c_denoised = self.eps_net(
            p_noisy, s_noisy, beta, mask_generate, mask_sample, linker_feat, pair_feat
        )   # (N, L, 3), (N, L, 3, 3), (N, L, 3), (N, L, 20), (N, L)

        loss_dict = {}

        # Position loss
        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction='none').sum(dim=-1)  # (N, L)
        loss_pos = (loss_pos * pos_mask).sum() / (pos_mask.sum().float() + 1e-8)
        loss_dict['atom_pos'] = loss_pos

        # Sequence categorical loss
        c_denoised = self.trans_type.before_softmax(c_denoised)
        loss_type = seq_cross_entropy(c_denoised, s_0_ignore)
        loss_type = (loss_type * type_mask).sum() / (type_mask.sum().float() + 1e-8)
        loss_dict['atom_type'] = loss_type

        return loss_dict
    
    @torch.no_grad()
    def sample(
        self, 
        p, s, 
        mask_generate, mask_sample, 
        sample_structure=True, 
        sample_type=True,
        pbar=False,
    ):
        """
        Args:
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
        """
        N, L = p.shape[:2]
        p = self._normalize_position(p)

        # Set the position of residues to be predicted to random values
        if sample_structure:
            p_rand = torch.randn_like(p)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
        else:
            p_init = p

        if sample_type:
            s_abosrb = torch.full_like(s, fill_value=num_total_type)
            s_init = torch.where(mask_generate, s_abosrb, s)
        else:
            s_init = s

        traj = {self.num_steps: (self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
        for t in pbar(range(self.num_steps, 0, -1)):
            p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            eps_p, c_denoised = self.eps_net(
                p_t, s_t, beta, mask_generate, mask_sample
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            c_denoised = self.trans_type.before_softmax(c_denoised)
            s_next = self.trans_type.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = p_t
            if not sample_type:
                s_next = s_t

            traj[t-1] = (self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.
        (final_pos, final_s) = traj[0]
        return (final_pos, final_s), traj