from .pl_transforms import *
from .transform import *
from .hm_transforms import *


def get_transform(cfg):
    if cfg is None or len(cfg) == 0:
        return None
    tfms = []
    for t_dict in cfg:
        t_dict = copy.deepcopy(t_dict)
        cls = TRANSFORM_DICT[t_dict.pop('type')]
        tfms.append(cls(**t_dict))
    return Compose(tfms)
