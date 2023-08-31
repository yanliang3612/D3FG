import argparse
import torch
from utils.misc import *
from datasets import get_dataset
from utils.runner import Runner
from models import get_model
import numpy as np
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train/fg_linker_stage.yml')
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--log_dir', type=str, default='./experiments')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exp_name', type=str, default='stage_0412')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mixed_precision', type=str, choices=['no', 'fp16', 'bf16'], default='no')
    args = parser.parse_args()
    
    config, config_path = load_config(args)
    data, subsets, dataloaders = get_dataset(config.dataset, eval=args.eval)

    dataloader = dataloaders["train"]
    mean = []
    for batch in dataloader:
        batch_mean = (torch.logical_not((~batch['mask_heavyatom']).prod(dim=-1).bool()).sum()/batch['mask_heavyatom'].shape[0]).item()
        mean.append(batch_mean)
    print(np.mean(mean))