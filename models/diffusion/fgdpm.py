import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm
from ..utils.so3 import so3vec_to_rotation, random_uniform_so3
from .transition import RotationTransition, PositionTransition, PLTypeTransition
from datasets.molecule.constants import num_fg_types, num_atom_types
from datasets.protein.constants import num_aa_types
from .ganet import GANet
from .loss import *

num_total_type = num_fg_types + num_aa_types + num_atom_types

class FGDPM(nn.Module):

    def __init__(
        self, 
        fg_feat_dim, 
        pair_feat_dim, 
        num_steps, 
        eps_net_opt={'type_out_num': num_total_type, 'use_egnn_update':True}, 
        trans_rot_opt={}, 
        trans_pos_opt={}, 
        trans_type_opt={
            'min_type_num':num_aa_types + num_atom_types, 
            'max_type_num':num_aa_types + num_atom_types + num_fg_types, 
            'num_classes':num_total_type
            },
        position_mean=[0.0, 0.0, 0.0],
        position_scale=10.0,
    ):
        super().__init__()
        self.eps_net = GANet(fg_feat_dim, pair_feat_dim, **eps_net_opt)
        self.num_steps = num_steps
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
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
        v_0, 
        p_0, 
        s_0, 
        fg_feat, 
        pair_feat, 
        mask_generate, 
        mask_sample, 
        denoise_structure, 
        denoise_type, 
        t=None
        ):

        batch_size = fg_feat.shape[0]

        if t is None:
            t = torch.randint(0, self.num_steps, (batch_size,), dtype=torch.long, device=self._dummy.device)
        elif len(t.shape) == 0:
            t = t.repeat(batch_size)

        p_0 = self._normalize_position(p_0)

        if denoise_structure:
            # Add noise to rotation
            R_0 = so3vec_to_rotation(v_0)
            v_noisy, _, rot_mask = self.trans_rot.add_noise(v_0, mask_generate, t, consider_single=True)
            # Add noise to positions
            p_noisy, eps_p, pos_mask = self.trans_pos.add_noise(p_0, mask_generate, t)
        else:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy = v_0.clone()
            p_noisy = p_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        if denoise_type:
            # Add noise to sequence
            s_0_ignore, s_noisy, type_mask = self.trans_type.add_noise(s_0, mask_generate, t)
        else:
            s_noisy = s_0.clone()

        beta = self.trans_pos.var_sched.betas[t]
        v_pred, R_pred, eps_p_pred, c_denoised = self.eps_net(
            v_noisy, p_noisy, s_noisy, fg_feat, pair_feat, beta, mask_generate, mask_sample
        )   # (N, L, 3), (N, L, 3, 3), (N, L, 3), (N, L, 20), (N, L)

        loss_dict = {}

        # Rotation loss
        loss_rot = rotation_matrix_cosine_loss(R_pred, R_0) # (N, L)
        loss_rot = (loss_rot * rot_mask).sum() / (rot_mask.sum().float() + 1e-8)
        loss_dict['fg_rot'] = loss_rot

        # Position loss
        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction='none').sum(dim=-1)  # (N, L)
        loss_pos = (loss_pos * pos_mask).sum() / (pos_mask.sum().float() + 1e-8)
        loss_dict['fg_pos'] = loss_pos

        # Sequence categorical loss
        c_denoised = self.trans_type.before_softmax(c_denoised)
        loss_type = seq_cross_entropy(c_denoised, s_0_ignore)
        loss_type = (loss_type * type_mask).sum() / (type_mask.sum().float() + 1e-8)
        loss_dict['fg_type'] = loss_type

        return loss_dict

    @torch.no_grad()
    def generate_single_atom_mask(self, single_atom_sampler, mask_sample, mask_generate):
        N, L = mask_sample.shape

        if single_atom_sampler is not None:
            single_atom_ratio = single_atom_sampler.sample(N)
        else:
            single_atom_ratio = (torch.rand(N) + 0.8)/1.8

        dummy_num = L - mask_sample.sum(dim=-1)
        generate_num = mask_generate.sum(dim=-1)
        v_zero_num = (generate_num * single_atom_ratio.to(generate_num.device)).long()
        v_nonzero_num = L - dummy_num - v_zero_num

        mask_v_nonzero = torch.le(
            torch.arange(0, L).unsqueeze(dim=0).repeat(N,1).to(generate_num.device), 
            v_nonzero_num.unsqueeze(dim=1).repeat(1, L)
            )
        
        mask_v_nonzero = torch.logical_and(mask_v_nonzero, mask_generate)

        return mask_v_nonzero

    @torch.no_grad()
    def sample(
        self, 
        v, p, s, 
        fg_feat, pair_feat, 
        mask_generate, mask_sample, 
        mask_wo_v=True,
        sample_structure=True, 
        sample_type=True,
        pbar=False,
        use_old_pos=False
    ):
        """
        Args:
            v:  Orientations of contextual residues, (N, L, 3).
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)
        
        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=self._dummy.device)
            mask_generate_v = mask_wo_v * mask_generate
            v_init = torch.where(mask_generate_v[:, :, None].expand_as(v), v_rand, v)

            p_rand = torch.randn_like(p)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
        else:
            v_init, p_init = v, p

        if sample_type:
            s_abosrb = torch.full_like(s, fill_value=num_total_type)
            s_init = torch.where(mask_generate, s_abosrb, s)
        else:
            s_init = s

        traj = {self.num_steps: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
        for t in pbar(range(self.num_steps, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, fg_feat, pair_feat, beta, mask_generate, mask_sample
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate_v, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            
            if use_old_pos:
                p_next = p

            c_denoised = self.trans_type.before_softmax(c_denoised)
            s_next = self.trans_type.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_type:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.
        (final_v, final_pos, final_s) = traj[0]
        return (final_v, final_pos, final_s), traj

    @torch.no_grad()
    def optimize(
        self, 
        v, p, s, 
        opt_step: int,
        fg_feat, pair_feat, 
        mask_generate, mask_sample, 
        sample_structure=True, sample_type=True,
        pbar=False,
    ):
        """
        Description:
            First adds noise to the given structure, then denoises it.
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)
        t = torch.full([N, ], fill_value=opt_step, dtype=torch.long, device=self._dummy.device)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            # Add noise to rotation
            v_noisy, _ = self.trans_rot.add_noise(v, mask_generate, t)
            # Add noise to positions
            p_noisy, _ = self.trans_pos.add_noise(p, mask_generate, t)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_noisy, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_noisy, p)
        else:
            v_init, p_init = v, p

        if sample_type:
            _, s_noisy = self.trans_type.add_noise(s, mask_generate, t)
            s_init = torch.where(mask_generate, s_noisy, s)
        else:
            s_init = s

        traj = {opt_step: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=opt_step, desc='Optimizing')
        else:
            pbar = lambda x: x
        for t in pbar(range(opt_step, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, fg_feat, pair_feat, beta, mask_generate, mask_sample
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_type.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_type:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj
