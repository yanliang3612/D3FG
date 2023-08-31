
import argparse
import torch

from .misc import *
from datasets import get_dataset
import torch.nn as nn 

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import numpy as np
from torch.distributions.categorical import Categorical
import pickle

class PreGenSize(nn.Module):
        def __init__(self, size_freq, size_prob) -> None:
            super().__init__()
        
            self.class_num = len(size_freq)
            self.prob = torch.tensor([v for k,v in size_prob.items()], requires_grad=False)
            self.type_to_num = {
                i:k for i,k in zip(range(self.class_num), size_freq.keys())
                }
            self.num_to_type = {
                v:k for k,v in self.type_to_num.items()
                }
            self.num_table = torch.tensor([v for k,v in self.type_to_num.items()], requires_grad=False)
            self.prior_categorical_sampler = Categorical(probs=self.prob)

        def sample(self, size, *args, **kwargs):
            return self.num_table[self.prior_categorical_sampler.sample((size,))]
        
        def forward(self, size, *args, **kwargs):
            return self.sample(size)

def prepare_pretrain_model(args):
    with open(args.pregen_path + 'linker_size_freq.pkl', 'rb') as f:
        linker_size_freq = pickle.load(f)
    with open(args.pregen_path + 'fg_size_freq.pkl', 'rb') as f:
        fg_size_freq = pickle.load(f)
    with open(args.pregen_path + 'linker_size_prob.pkl', 'rb') as f:
        linker_size_prob = pickle.load(f)
    with open(args.pregen_path + 'fg_size_prob.pkl', 'rb') as f:
        fg_size_prob = pickle.load(f)

    size_pregen_linker = PreGenSize(linker_size_freq, linker_size_prob)

    size_pregen_fg = PreGenSize(fg_size_freq, fg_size_prob)

    return size_pregen_linker, size_pregen_fg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train/functional_group_based.yml')
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--log_dir', type=str, default='./experiments')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp_name', type=str, default='fg_linker_diff')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mixed_precision', type=str, choices=['no', 'fp16', 'bf16'], default='no')
    parser.add_argument('--pregen_path', type=str, default='./scripts/size_pregen_model/')
    parser.add_argument('--force_to_prepare', type=bool, default=False)
    args = parser.parse_args()

    config, config_path = load_config(args)
    if args.force_to_prepare:
        data, subsets, dataloaders = get_dataset(config.dataset, eval=args.eval)
        data_loader = dataloaders["train"]
        linker_sizes = []
        fg_sizes = []

        for batch in data_loader:
            linker_size = batch['linker_size']
            fg_size = batch['fg_size']
            linker_sizes.append(linker_size)
            fg_sizes.append(fg_size)

        linker_sizes = torch.cat(linker_sizes).numpy()
        fg_sizes = torch.cat(fg_sizes).numpy()

        linker_size_uniq, count_linker = np.unique(linker_sizes, return_counts=True)
        fg_size_uniq, count_fg = np.unique(fg_sizes, return_counts=True)
        prob_linker = count_linker/len(linker_sizes)
        prob_fg = count_fg/len(fg_sizes)

        linker_size_freq = dict(zip(linker_size_uniq, count_linker))
        fg_size_freq = dict(zip(fg_size_uniq, count_fg))

        linker_size_prob = dict(zip(linker_size_uniq, prob_linker))
        fg_size_prob = dict(zip(fg_size_uniq, prob_fg))

        with open(args.pregen_path + 'linker_size_freq.pkl', 'wb') as f:
            pickle.dump(linker_size_freq, f)

        with open(args.pregen_path + 'fg_size_freq.pkl', 'wb') as f:
            pickle.dump(fg_size_freq, f)

        with open(args.pregen_path + 'linker_size_prob.pkl', 'wb') as f:
            pickle.dump(linker_size_prob, f)

        with open(args.pregen_path + 'fg_size_prob.pkl', 'wb') as f:
            pickle.dump(fg_size_prob, f)

    else:
        with open(args.pregen_path + 'linker_size_freq.pkl', 'rb') as f:
            linker_size_freq = pickle.load(f)
        with open(args.pregen_path + 'fg_size_freq.pkl', 'rb') as f:
            fg_size_freq = pickle.load(f)
        with open(args.pregen_path + 'linker_size_prob.pkl', 'rb') as f:
            linker_size_prob = pickle.load(f)
        with open(args.pregen_path + 'fg_size_prob.pkl', 'rb') as f:
            fg_size_prob = pickle.load(f)

    size_pregen_linker = PreGenSize(linker_size_freq, linker_size_prob)
    size_pregen_linker.sample(32)

    size_pregen_fg = PreGenSize(fg_size_freq, fg_size_prob)



