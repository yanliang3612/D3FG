from .transforms import get_transform
import torch
from torch.utils.data import Subset, DataLoader
from .utils import *

_DATASET_DICT = {}


def register_dataset(name):
    def decorator(cls):
        _DATASET_DICT[name] = cls
        return cls
    return decorator

def get_dataset(cfg, eval):
    if eval:
        cfg.transform.append({'type':'remove_gen'})
    transform = get_transform(cfg.transform) if 'transform' in cfg else None
    dataset = _DATASET_DICT[cfg.type](cfg, transform=transform)
    split_by_name = torch.load(cfg.split_path)
    split = {
        k: [dataset.name2id[n] for n in names if n in dataset.name2id]
        for k, names in split_by_name.items()
    }

    subsets = {k: (Subset(dataset, indices=v)) for k, v in split.items()}

    train_set, val_set = subsets["train"], subsets["test"]

    train_loader = DataLoader(
                train_set,
                batch_size=cfg.batch_size,
                shuffle=True,
                collate_fn=PaddingCollate(),
            )
    val_loader = DataLoader(
                val_set,
                cfg.batch_size,
                shuffle=False,
                collate_fn=PaddingCollate(),
            )

    dataloaders = {"train": train_loader, "val": val_loader, "test": val_loader}
        
    return dataset, subsets, dataloaders
