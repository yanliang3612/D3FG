from easydict import EasyDict
import yaml
import os
import time

def load_config(args):
    config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    config.update(vars(args))
    
    return config, config_path


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self

def current_milli_time():
    return round(time.time() * 1000)

def log_losses(out, epoch, tag, logger=BlackHole(), writer=BlackHole(), others={}):
    logstr = '[%s] Epoch %05d' % (tag, epoch)
    logstr += ' | loss %.4f' % out['overall'].item()
    for k, v in out.items():
        if k == 'overall': continue
        logstr += ' | loss(%s) %.4f' % (k, v.item())
    for k, v in others.items():
       logstr += ' | %s %2.4f' % (k, v)
    logger.info(logstr)

    for k, v in out.items():
        if k == 'overall':
            writer.add_scalar('%s/loss' % tag, v, epoch)
        else:
            writer.add_scalar('%s/loss_%s' % (tag, k), v, epoch)
    for k, v in others.items():
        writer.add_scalar('%s/%s' % (tag, k), v, epoch)
    writer.flush()