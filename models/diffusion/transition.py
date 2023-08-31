import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.layers import clampped_one_hot
from ..utils.so3 import ApproxAngularDistribution, random_normal_so3, so3vec_to_rotation, rotation_to_so3vec
from ..utils.misc import *
from datasets.molecule.constants import num_fg_types
from datasets.protein.constants import num_aa_types

num_total_type = num_fg_types + num_aa_types

class VarianceSchedule(nn.Module):

    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        T = num_steps
        t = torch.arange(0, num_steps+1, dtype=torch.float)
        f_t = torch.cos( (np.pi / 2) * ((t/T) + s) / (1 + s) ) ** 2
        alpha_bars = f_t / f_t[0]

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        betas = betas.clamp_max(0.999)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('sigmas', sigmas)

class TypeMasker(nn.Module):
    def __init__(
        self, 
        num_steps=100, 
        mask_id=num_total_type, 
        unmasked_steps=0, 
        noise_schedule='uniform'
        ) -> None:
        super().__init__()
        self.noise_shedule = noise_schedule
        self.num_steps = num_steps
        self.unmasked_steps = unmasked_steps
        self.mask_id = mask_id
        self.register_buffer('_dummy', torch.empty([0, ]))


    def forward(self, x_0, t, node_mask=1, eps=None):

        node_mask = node_mask.bool()
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        if eps is not None:
            mask_prob = eps  
        else:
            mask_prob = (
                (t.view(-1).float()-self.unmasked_steps).clamp(min=0.)
                / (self.num_steps-self.unmasked_steps)
                ).to(self._dummy.device)

        diff_mask = (
            torch.rand_like(x_t.float()) < inflate_batch_array(mask_prob, x_t.float())
            )

        # if true, the element will be masked
        diff_mask = torch.logical_and(diff_mask, node_mask)
        x_t[diff_mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(diff_mask)] = -1
        return x_t, x_0_ignore, diff_mask, mask_prob
    
    def reveil_mask(self, x_disc, t):
        prob = (
            (t-self.num_steps) / (self.unmasked_steps-self.num_steps)
            ).clamp(max=1., min=0.)
        changes = (
            torch.rand_like(x_disc.float()) 
            < inflate_batch_array(prob, x_disc)
        )
        return changes



class PositionTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    def add_noise(self, p_0, mask_generate, t):
        """
        Args:
            p_0:    (N, L, 3).
            mask_generate:    (N, L).
            t:  (N,).
        """
        alpha_bar = self.var_sched.alpha_bars[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        e_rand = torch.randn_like(p_0)
        p_noisy = c0*p_0 + c1*e_rand
        p_noisy = torch.where(mask_generate[..., None].expand_as(p_0), p_noisy, p_0)

        return p_noisy, e_rand, mask_generate

    def denoise(self, p_t, eps_p, mask_generate, t):
        # IMPORTANT:
        #   clampping alpha is to fix the instability issue at the first step (t=T)
        #   it seems like a problem with the ``improved ddpm''.
        alpha = self.var_sched.alphas[t].clamp_min(
            self.var_sched.alphas[-2]
        )
        alpha_bar = self.var_sched.alpha_bars[t]
        sigma = self.var_sched.sigmas[t].view(-1, 1, 1)

        c0 = ( 1.0 / torch.sqrt(alpha + 1e-8) ).view(-1, 1, 1)
        c1 = ( (1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8) ).view(-1, 1, 1)

        z = torch.where(
            (t > 1)[:, None, None].expand_as(p_t),
            torch.randn_like(p_t),
            torch.zeros_like(p_t),
        )

        p_next = c0 * (p_t - c1 * eps_p) + sigma * z
        p_next = torch.where(mask_generate[..., None].expand_as(p_t), p_next, p_t)
        return p_next


class RotationTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}, angular_distrib_fwd_opt={}, angular_distrib_inv_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

        # Forward (perturb)
        c1 = torch.sqrt(1 - self.var_sched.alpha_bars) # (T,).
        self.angular_distrib_fwd = ApproxAngularDistribution(c1.tolist(), **angular_distrib_fwd_opt)

        # Inverse (generate)
        sigma = self.var_sched.sigmas
        self.angular_distrib_inv = ApproxAngularDistribution(sigma.tolist(), **angular_distrib_inv_opt)

        self.register_buffer('_dummy', torch.empty([0, ]))

    def add_noise(self, v_0, mask_generate, t, consider_single=True):
        """
        Args:
            v_0:    (N, L, 3).
            mask_generate:    (N, L).
            t:  (N,).
        """
        if consider_single:
            mask_single = torch.logical_not(torch.sum(v_0, dim=-1).abs() < 1e-6)
        else:
            mask_single = True

        mask_generate = mask_single * mask_generate
        N, L = mask_generate.size()
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        # Noise rotation
        e_scaled = random_normal_so3(t[:, None].expand(N, L), self.angular_distrib_fwd, device=self._dummy.device)    # (N, L, 3)
        e_normal = e_scaled / (c1 + 1e-8)
        E_scaled = so3vec_to_rotation(e_scaled)   # (N, L, 3, 3)

        # Scaled true rotation
        R0_scaled = so3vec_to_rotation(c0 * v_0)  # (N, L, 3, 3)

        R_noisy = E_scaled @ R0_scaled
        v_noisy = rotation_to_so3vec(R_noisy)
        v_noisy = torch.where(mask_generate[..., None].expand_as(v_0), v_noisy, v_0)

        return v_noisy, e_scaled, mask_generate

    def denoise(self, v_t, v_next, mask_generate, t):
        N, L = mask_generate.size()
        e = random_normal_so3(t[:, None].expand(N, L), self.angular_distrib_inv, device=self._dummy.device) # (N, L, 3)
        e = torch.where(
            (t > 1)[:, None, None].expand(N, L, 3),
            e, 
            torch.zeros_like(e) # Simply denoise and don't add noise at the last step
        )
        E = so3vec_to_rotation(e)

        R_next = E @ so3vec_to_rotation(v_next)
        v_next = rotation_to_so3vec(R_next)
        v_next = torch.where(mask_generate[..., None].expand_as(v_next), v_next, v_t)

        return v_next

class PLTypeTransition(nn.Module):
    
    def __init__(
        self, 
        num_steps, 
        min_type_num, 
        max_type_num, 
        num_classes,
        var_sched_opt={'noise_schedule': 'uniform'}):
        super().__init__()
        self.num_classes = num_classes
        self.masker =  TypeMasker(
            num_steps, 
            mask_id=self.num_classes, 
            unmasked_steps=num_steps//2,
            **var_sched_opt
            )

        self.min_type_num = min_type_num
        self.max_type_num = max_type_num
        self.register_buffer('_dummy', torch.empty([0, ]))


    def sample(self, c):
        """
        Args:
            c:    (N, L, K).
        Returns:
            x:    (N, L).
        """
        N, L, K = c.size()
        c = self.logits2prob(c)
        c = c.view(N*L, K) + 1e-8
        x = torch.multinomial(c, 1).view(N, L)
        return x


    def add_noise(self, x_0, mask_generate, t, eps=None):
        """
        Args:
            x_0:    (N, L)
            mask_generate:    (N, L).
            t:  (N,).
        Returns:
            x_0_ignore:    Truth with mask, LongTensor, (N, L).
            x_t:    Sample, LongTensor, (N, L).
            type_mask: Mask for calculate the loss, (N, L).
        """

        x_t, x_0_ignore, type_mask, __ = self.masker(x_0, t, mask_generate, eps)

        return x_0_ignore, x_t, type_mask

    
    def before_softmax(self, x):
        assert x.dim() == 3
        N, L, M = x.size()

        mask_type = torch.zeros_like(x).bool()
        mask_type[:,:,:self.min_type_num] = True
        mask_type[:,:,self.max_type_num:] = True

        logits_pred = torch.where(mask_type, x-1e8, x)

        return logits_pred


    def logits2prob(self, x):
        logits_pred = self.before_softmax(x)
        x_pred = F.softmax(logits_pred, dim=-1)
        return x_pred


    def denoise(self, x_t, c_0_pred, mask_generate, t, temp=1.0):
        """
        Args:
            x_t:        (N, L).
            c_0_pred:   Normalized probability predicted by networks, (N, L, K).
            mask_generate:    (N, L).
            t:  (N,).
        Returns:
            post:   Posterior probability at (t-1)-th step, (N, L, K).
            x_next: Sample at (t-1)-th step, LongTensor, (N, L).
        """

        changes = self.masker.reveil_mask(x_disc=x_t, t=t-1)
        unmasked = torch.logical_or((x_t!=self.masker.mask_id), (~mask_generate).bool())
        changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))

        x_logits = c_0_pred / temp
        x_next = self.sample(x_logits)

        x_t[changes] = x_next[changes]

        return x_t


class AminoacidCategoricalTransition(nn.Module):
    
    def __init__(self, num_steps, num_classes=20, var_sched_opt={}):
        super().__init__()
        self.num_classes = num_classes
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    @staticmethod
    def _sample(c):
        """
        Args:
            c:    (N, L, K).
        Returns:
            x:    (N, L).
        """
        N, L, K = c.size()
        c = c.view(N*L, K) + 1e-8
        x = torch.multinomial(c, 1).view(N, L)
        return x

    def add_noise(self, x_0, mask_generate, t):
        """
        Args:
            x_0:    (N, L)
            mask_generate:    (N, L).
            t:  (N,).
        Returns:
            c_t:    Probability, (N, L, K).
            x_t:    Sample, LongTensor, (N, L).
        """
        N, L = x_0.size()
        K = self.num_classes
        c_0 = clampped_one_hot(x_0, num_classes=K).float() # (N, L, K).
        alpha_bar = self.var_sched.alpha_bars[t][:, None, None] # (N, 1, 1)
        c_noisy = (alpha_bar*c_0) + ( (1-alpha_bar)/K )
        c_t = torch.where(mask_generate[..., None].expand(N,L,K), c_noisy, c_0)
        x_t = self._sample(c_t)
        return c_t, x_t

    def posterior(self, x_t, x_0, t):
        """
        Args:
            x_t:    Category LongTensor (N, L) or Probability FloatTensor (N, L, K).
            x_0:    Category LongTensor (N, L) or Probability FloatTensor (N, L, K).
            t:  (N,).
        Returns:
            theta:  Posterior probability at (t-1)-th step, (N, L, K).
        """
        K = self.num_classes

        if x_t.dim() == 3:
            c_t = x_t   # When x_t is probability distribution.
        else:
            c_t = clampped_one_hot(x_t, num_classes=K).float() # (N, L, K)

        if x_0.dim() == 3:
            c_0 = x_0   # When x_0 is probability distribution.
        else:
            c_0 = clampped_one_hot(x_0, num_classes=K).float() # (N, L, K)

        alpha = self.var_sched.alpha_bars[t][:, None, None]     # (N, 1, 1)
        alpha_bar = self.var_sched.alpha_bars[t][:, None, None] # (N, 1, 1)

        theta = ((alpha*c_t) + (1-alpha)/K) * ((alpha_bar*c_0) + (1-alpha_bar)/K)   # (N, L, K)
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        return theta

    def denoise(self, x_t, c_0_pred, mask_generate, t):
        """
        Args:
            x_t:        (N, L).
            c_0_pred:   Normalized probability predicted by networks, (N, L, K).
            mask_generate:    (N, L).
            t:  (N,).
        Returns:
            post:   Posterior probability at (t-1)-th step, (N, L, K).
            x_next: Sample at (t-1)-th step, LongTensor, (N, L).
        """
        c_t = clampped_one_hot(x_t, num_classes=self.num_classes).float()  # (N, L, K)
        post = self.posterior(c_t, c_0_pred, t=t)   # (N, L, K)
        post = torch.where(mask_generate[..., None].expand(post.size()), post, c_t)
        x_next = self._sample(post)
        return post, x_next
