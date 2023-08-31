
import argparse
import torch

from utils.misc import *
from datasets import get_dataset
from utils.runner import Runner
from models import get_model
from utils.fg_linker_size_stats import prepare_pretrain_model
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/generate/fg_linker_stage.yml')
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--log_dir', type=str, default='./experiments')
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--exp_name', type=str, default='stage_0412')
    parser.add_argument('--mixed_precision', type=str, choices=['no', 'fp16', 'bf16'], default='no')
    parser.add_argument('--pregen_path', type=str, default='./scripts/size_pregen_model/')

    args = parser.parse_args()
    
    config, config_path = load_config(args)
    data, subsets, dataloaders = get_dataset(config.dataset, eval=args.eval)

    model = get_model(
        config.model, 
        device=args.device
        )
    
    trainer = Runner(
        data=dataloaders, 
        model=model,
        config_path=config_path, 
        config=config,
        epoch_num=4000
        )
    
    pregen_linker, pregen_fg = prepare_pretrain_model(args)
    

    trainer.generate(pregen_model = {'pregen_linker':pregen_linker, 'pregen_fg':pregen_fg})
    