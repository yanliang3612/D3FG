import logging
import os
import random
import time
import warnings
from .misc import *
import numpy as np
import torch


class Queue:
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)

    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def gradient_clipping(model, gradnorm_queue, verbose=False, clip=torch.nn.utils):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = clip.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm, norm_type=2.0
    )

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm and verbose:
        print(
            f"Clipped gradient with value {grad_norm:.1f} "
            f"while allowed {max_grad_norm:.1f}"
        )
    return grad_norm


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def get_optimizer(cfg, model):
    if cfg.type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(
                cfg.beta1,
                cfg.beta2,
            ),
        )
    elif cfg.type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            amsgrad=True,
            weight_decay=cfg.weight_decay,
            betas=(
                cfg.beta1,
                cfg.beta2,
            ),
        )

    else:
        raise NotImplementedError("Optimizer not supported: %s" % cfg.type)


def get_scheduler(cfg, optimizer):
    if cfg.type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == "expmin":
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=cfg.factor,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == "expmin_milestone":
        gamma = np.exp(np.log(cfg.factor) / cfg.milestone)
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=gamma,
            min_lr=cfg.min_lr,
        )
    else:
        raise NotImplementedError("Scheduler not supported: %s" % cfg.type)


class ExponentialLR_with_minLr(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr=1e-4, last_epoch=-1, verbose=False):
        self.gamma = gamma
        self.min_lr = min_lr
        super(ExponentialLR_with_minLr, self).__init__(
            optimizer, gamma, last_epoch, verbose
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return self.base_lrs
        return [
            max(group["lr"] * self.gamma, self.min_lr)
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]


def get_logger(name, log_dir=None, log_fn="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s::%(name)s::%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, log_fn))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_log_dir(root="./logs", prefix="", tag="", exp_name=None):
    fn = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    if exp_name is not None:
        fn = str(exp_name)
    else:
        fn = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    if prefix != "":
        fn = prefix + "_" + fn
    if tag != "":
        fn = fn + "_" + tag
    log_dir = os.path.join(root, fn)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


class ValidationLossTape(object):

    def __init__(self, logger, writer):
        super().__init__()
        self.accumulate = {}
        self.others = {}
        self.total = 0
        self.logger = logger
        self.writer = writer

    def update(self, out, n, others={}):
        self.total += n
        for k, v in out.items():
            if k not in self.accumulate:
                self.accumulate[k] = v.clone().detach()
            else:
                self.accumulate[k] += v.clone().detach()

        for k, v in others.items():
            if k not in self.others:
                self.others[k] = v.clone().detach()
            else:
                self.others[k] += v.clone().detach()
        

    def log(self, it, tag='val'):
        avg = EasyDict({k:v / self.total for k, v in self.accumulate.items()})
        avg_others = EasyDict({k:v / self.total for k, v in self.others.items()})
        log_losses(avg, it, tag, self.logger, self.writer, others=avg_others)
        return avg['overall']
    